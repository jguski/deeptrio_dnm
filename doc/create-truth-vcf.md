# create-truth-vcfs

|Script:|../scripts/utils/create_truth_vcf.py|
|:--:|:--:|
|Environment:|../envs/data_augmentation.yml|


We use `create_truth_vcf.py` to annotate a vcf file with truth information
from a reference. In the case of our study, we use the overlap of multiple
variant callers as gold standard. The parameters `--dnm1` and `--dnm2` and
`--msdn1` and `--msdn2` respectively are used to flag the callsets for the two
variant classes and each caller.

To generate the `vcf` and `bed` files required by the subsequent steps, run:

```bash
python3 scripts/utils/create_truth_vcf.py \
    --index_id ${sample} \
    --vcf ${input_vcf} \
    --dnm1 ${first_caller_vcf} \
    --dnm2 ${second_caller_vcf} \
    --msdn1 ${first_caller_vcf} \
    --msdn2 ${second_caller_vcf} \
    --prop_std_classes 1 \
    --out_dir $TMPDIR

# Copy and Format output VCF file
mkdir -p ${out_dir}
bgzip -c $TMPDIR/${sample}.vcf > ${output_vcf}
tabix ${output_vcf}

# Create Output BED file
bcftools query -f'%CHROM\\t%POS0\\t%END\\n' ${output_vcf} > ${bed_out}
```

## Second training round

During retraining, we noticed some false positive _de novo_ mutation calls
and generated training examples to specifically adress these classes of variants.

To generate the truth vcf, additionally supply the options

```bash
--indel_window_file ${path}
--first_round_file ${path}
```

to `create_truth_vcf.py`.

### Get variants around InDels

Here `indel_window_file` should be the path to a vcf
file containing variants up- and downstream of indels in a 110bp window. It 
can be created using the following `bcftools` commands:

```bash
prefix=$(mktemp -d)
bcftools view -i "FILTER='PASS'" | \
    bgzip -@${threads} > ${prefix}.full.vcf.bgz
tabix ${prefix}.full.vcf.bgz

bcftools filter ${prefix}.full.vcf.bgz -g 110 | \
    bgzip -@${threads} > ${prefix}.filtered.vcf.bgz
tabix ${prefix}.filtered.vcf.bgz

bcftools isec -p $TMPDIR \
    ${prefix}.full.vcf.bgz \
    ${prefix}.filtered.vcf.bgz

# Copy and Format output VCF file
mkdir -p $(dirname ${output_vcf})
bgzip -c $TMPDIR/0000.vcf > ${output_vcf}
tabix ${output_vcf}
```

### Get False-Positive variants after the first round of training

The `first_round_file` can be generated with the help of the script `check_false_positive.py`. It uses the parental allele frequency to identify putative _de novo_ variants that are likely inherited and the index allele frequency to detect likely events of homozygous reference sites being misclassified.

To run the script, execute the following commands. `input_vcf` should be a 
vcf generated by `create_truth_vcfs.py` for the first round of training,
`call_table` the output of the network after the first round of retraining and
`trio_vcf` the original VCF for the trio that is used for the retraining. `sample` denotes the index sample id in the vcf.

```bash
python3 check_false_positive.py \
    --truth_vcf ${input_vcf} \
    --call_table ${call_table} \
    --trio_vcf ${trio_vcf} \
    --sample ${sample} | bcftools sort -O z -o ${output_vcf}

tabix ${output_vcf}
bcftools query -f '%CHROM\\t%POS0\\t%END\\n' ${output_vcf} > ${output_bed}
```