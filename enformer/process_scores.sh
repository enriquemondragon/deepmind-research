#!/bin/bash
file=sad_scores.csv #EDIT THIS
IFS=. read -r name ext <<< ${file};

sed -e :a -e '$!N;s/\n"/ /;ta' -e 'P;D' ${file} > temp.vcf
sed "s/\"//g" temp.vcf > temp_2.vcf
sed -e 's/ \t/\t/g' temp_2.vcf > ${name}_2.vcf
rm temp*

