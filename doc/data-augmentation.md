# data-augmentation

|Script:|../scripts/data_augmentation.py|
|:--:|:--:|
|Environment:|../envs/data_augmentation.yml|

Since true CDNM calls are very rare in normal samples, we decided to augment the training data by generating synthetic clusters. Synthetic clusters are generated in proximity to real variants and regions that fulfill criteria (coverage etc.). Then we augment the read files, s.t. the cluster should be detectable by the models. Additionally, some false positive training examples are generated. Genereated clusters vary in their size, and how big the distances between individual lesions is.

As input, we require the original bam files for the trio and the original vcf.

Together with `samtools` and `bcftools`, the script can be run:

```bash
TMPDIR=$(mktemp -d)
python3 data_augmentation.py \
  --original_vcf ${input_vcf} \
  --original_bam_child ${input_child_bam} \
  --original_bam_parent1 ${input_father_bam} \
  --original_bam_parent2 ${input_mother_bam} \
  --output_directory $TMPDIR

samtools sort $TMPDIR/synthetic_MSDN_child_unsorted.bam -o ${output_child_bam}
samtools index ${output_child_bam}
samtools sort $TMPDIR/synthetic_MSDN_parent1_unsorted.bam -o ${output_father_bam}
samtools index ${output_father_bam}
samtools sort $TMPDIR/synthetic_MSDN_parent2_unsorted.bam -o ${output_mother_bam}
samtools index ${output_mother_bam}

bcftools sort -O z -o ${output_vcf} $TMPDIR/synthetic_MSDN_unsorted.vcf
bcftools index -t ${output_vcf}

bcftools query -f'%CHROM\t%POS0\t%END\n' ${output_vcf} > ${output_bed}
```