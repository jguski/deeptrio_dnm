# make-examples

The make examples step to generate training data for the network has not been changed from the original DeepTrio. Therefore, we assume that a singularity container of deepvariant with DeepTrio is installed. Then, the following commands can be used to generate a set of examples for retraining:

Here, the following inputs are used:

  * `truth_vcf` and `truth_vcf_bed`: Output from `create_truth_vcfs.py`, containing variant call information.
  * `classes_list`: The name of the classes to be used. Should be set as `0 1 2 3` for `HOM HET ALT DNM` and `0 1 2 3 4` for `HOM HET ALT DNM CDNM`, respectively.
  * When creating examples for variant calling and not retraining, `mode` can
  be changed to `calling` and all lines after and including `--regions` can
  be ommitted.

```bash
seq 0 ${threads} \
    | parallel --will-cite -P ${threads} --eta --halt 2 \
    singularity run --nv {deeptrio_container} /opt/deepvariant/bin/deeptrio/make_examples \
      --mode training
      --examples ${output_dir}/${base}.tfrecord@${threads}.gz \
      --ref ${reference} \
      --reads ${sample_bam} \
      --reads_parent1 ${father_bam} \
      --reads_parent2 ${mother_bam} \
      --sample_name ${sample_name} \
      --sample_name_parent1 ${father_name} \
      --sample_name_parent2 ${mother_name} \
      --task {} \
      --regions ${truth_vcf_bed} \
      --truth_variants ${truth_vcf} \
      --labeler_algorithm customized_classes_labeler \
      --customized_classes_labeler_info_field_name "label" \
      --customized_classes_labeler_classes_list ${classes_list} \
      --vsc_min_fractions_snps 0 \
      --vsc_min_count_snps 0

# Create a dataset descriptor for retraining / calling.
python3 utils/create_pbtxt.py \
    --sample_path ${output_dir} \
    --tfrecord_pattern "${base}.tfrecord-?????-of-?????.gz" \
    --dataset_config_pbtxt ${output_dir}/unshuffled.dataset_config.pbtxt
```