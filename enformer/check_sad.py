# check sad

# command: 
# python3 check_sad.py --input scores/debug/sad_scores.h5

from re import S
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description=' ################ SAD scores results ################', usage='%(prog)s')
parser.add_argument('-in', '--input', type=str, required=True, help='sad file', dest='sad_file')
args = parser.parse_args()
sad_file = args.sad_file

sad_h5 = h5py.File(sad_file, 'r')

print('\t',parser.description)
print('\nkeys are',list(sad_h5.keys()))

snps = sad_h5['scores'].shape
print('shape is', snps)
'''
print('\nNumber of targets:',num_targets)
print('Number of snps:',num_snps)


print('SAD scores are: \n',sad_h5['SAD'][:])

for ti in range(num_targets):
    print(ti,sad_h5['target_labels'][ti])
'''

