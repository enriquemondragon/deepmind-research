
import tensorflow as tf
import tensorflow_hub as hub
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import argparse
import sys
import joblib
import re
import h5py

# command
# python3 predict_scores.py -v data/CHANGE -f data/hg38.fa -t data/targets_human.txt -o outdir/

parser = argparse.ArgumentParser(description=' ################ Predict variant scores with Enformer ################', usage='%(prog)s')
parser.add_argument('-v', '--vcf_file', type=str, required=True, help='file with the genomic variants', dest='vcf_file')
parser.add_argument('-f', '--fasta_file', type=str, required=True, help='hg sequence file', dest='fasta_file')
parser.add_argument('-t', '--targets_file', type=str, required=True, help='targets file', dest='targets_file')
parser.add_argument('-o', '--out_dir', type=str, required=True, help='Output directory', dest='out_dir')

args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(tf.config.list_physical_devices('GPU'))

#############################################
# Loading data and models
#############################################

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1' # enformer = hub.Module('https://tfhub.dev/deepmind/enformer/1')
fasta_file = args.fasta_file
clinvar_vcf = args.vcf_file

targets_txt = args.targets_file # Cite: Kelley et al Cross-species regulatory sequence activity prediction. PLoS Comput. Biol. 16, e1008050 (2020).
df_targets = pd.read_csv(targets_txt, sep='\t')

pyfaidx.Faidx(fasta_file)


#############################################
# Enformer: Classes and functions
#############################################

SEQUENCE_LENGTH = 393216

class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

  def __init__(self, tfhub_url, organism='human'):
    self._model = Enformer(tfhub_url)
    self._organism = organism
  
  def predict_on_batch(self, inputs):
    ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
    alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

    return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human'):
    assert organism == 'human', 'Transforms only compatible with organism=human'
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      transform_pipeline = joblib.load(f)
    self._transform = transform_pipeline.steps[0][1]  # StandardScaler.
    
  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human', num_top_features=500):
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      self._transform = joblib.load(f)
    self._num_top_features = num_top_features
    
  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)[:, :self._num_top_features]


#############################################
# Fasta and variants: Classes and functions
#############################################

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
  """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
  def _open(file):
    return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    
  with _open(vcf_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      chrom, pos, id, ref, alt_list = line.split('\t')[:5]
      # Split ALT alleles and return individual variants as output.
      for alt in alt_list.split(','):
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                           ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
  seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

  for variant in variant_generator(vcf_file, gzipped=gzipped):
    print('\nVARIANT: ', variant)
    interval = Interval(chr_prefix + variant.chrom,
                        variant.pos, variant.pos)
    print('INTERVAL: ', interval)
    interval = interval.resize(sequence_length)
    center = interval.center() - interval.start
    print('INTERVAL: ', interval)

    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)

    print('REFERENCE (first 100bp): ', reference[0:100])
    print('ALTERNATE (first 100bp): ', alternate[0:100])
    print('REFERENCE (mid 100bp): ', reference[SEQUENCE_LENGTH//2-50:SEQUENCE_LENGTH//2+50])
    print('ALTERNATE (mid 100bp): ', alternate[SEQUENCE_LENGTH//2-50:SEQUENCE_LENGTH//2+50])
    print('REFERENCE (last 100bp): ', reference[SEQUENCE_LENGTH-100:SEQUENCE_LENGTH])
    print('ALTERNATE (last 100bp): ', alternate[SEQUENCE_LENGTH-100:SEQUENCE_LENGTH])
    print('num_spaces:',alternate.find('\n'))
    print('ONE HOT REF: ', one_hot_encode(reference))
    print('ONE HOT ALT: ', one_hot_encode(alternate))
    print('ONE HOT REF SHAPE: ', one_hot_encode(reference).shape)
    print('ONE HOT ALT SHAPE: ', one_hot_encode(alternate).shape)
    print('VARIANT REF AND ALT: ', variant.ref, variant.alt)
    
    yield {'inputs': {'ref': one_hot_encode(reference),
                      'alt': one_hot_encode(alternate)},
           'metadata': {'chrom': chr_prefix + variant.chrom,
                        'pos': variant.pos,
                        'id': variant.id,
                        'ref': variant.ref,
                        'alt': variant.alt}}


#############################################
# Score variants for 5313 targets
#############################################
enformer_score_variants_all = EnformerScoreVariantsNormalized(model_path, transform_path)

# Score the first 5 variants from ClinVar
# All Scores
it = variant_centered_sequences(clinvar_vcf, sequence_length=SEQUENCE_LENGTH,
                                gzipped=True, chr_prefix='chr')
example_list=[]
count = 0
for i, example in enumerate(it):
    print('Done: ', count)
    count = count + 1
    variant_scores = enformer_score_variants_all.predict_on_batch(
        {k: v[tf.newaxis] for k,v in example['inputs'].items()})[0]
    print('VARIANT SCORE: ', variant_scores, 'SHAPE:',variant_scores.shape)
    variant_scores = {f'{i}_{name[:20]}': score for i, (name, score) in enumerate(zip(df_targets.description, variant_scores))}
    example_list.append({**example['metadata'],
                       **variant_scores})

df = pd.DataFrame(example_list)
df
print('\nDATAFRAME IS: \n', df)

#############################################
# Write output file
#############################################

# cvs file
out_path = args.out_dir + 'sad_scores.csv'
df.to_csv (out_path, index = False, header=True, sep ='\t')

print('saved!')
