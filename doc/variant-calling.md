# variant-calling

|Script:|../scripts/calling.py|
|:--:|:--:|
|Environment:|../envs/tensorflow.yml|

The variant calling script uses a retrained model to create an output vcf and variant table (.tsv) for easier consumption by downstream processes. If only the table output is desired, the `calling_dnm.py` script can be used instead.

The following inputs are required to call variants on a set of examples:

 * `call_data_path`: Path to a folder with examples to call variants. These folders should by layed out identically to the ones used for retraining (see ./retraining.md)
 * `copy_header_from`: We copy the vcf header from any existing vcf, none of the variants or data from this vcf is used.
 * `classes`: Class names as string. Should be set to `"HOM" "HET" "ALT" "DNM" "MSDN"` if the CDNM model is used, and exclude `"MSDN"` if the DNM model is used.
 * `model`: A retrained model to use for variant calling.

To execute the script, run:

```bash
TMPDIR=$(mktemp -d)
python3 calling.py \
    --call_data_path ${call_data_path} \
    --model_path ${model} \
    --copy_header_from ${copy_header_from_vcf} \
    --output_vcf $TMPDIR/output.vcf \
    --output_table ${output_table} \
    --batch_size 128 \
    --classes "HOM" "HET" "ALT" "DNM" "MSDN" \
    --dnm_threshold 0.9 \
    --msdn_threshold 0.9

bcftools sort -O z -o ${output_vcf} $TMPDIR/output.vcf
tabix ${output_vcf}
```