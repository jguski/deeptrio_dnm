import os
import glob
import sys
from re import split
from absl import logging
import math
import numpy as np
import tensorflow as tf
from .create_pbtxt import create_pbtxt
from utils.protos import deepvariant_pb2
from utils.protos.third_party.nucleus.protos import variants_pb2


def _phred_score(x, default=1000):
    if x > 0:
        return -10 * math.log10(x)
    else:
        return default


def parse_tfexample(tf_example, input_shape, n_classes, calling, combine_labels=None):
    """Parses a given an example loaded from a tfrecord file
    Args:
      tf_example: Example loaded from tfrecord file as created in the make_examples step of DeepTrio.
      input_shape: Shape of input tensors
      n_classes: Number of predicted classes
      calling: If in calling mode, not label but other features will be read from tfrecord file

    Returns:
      Pileup image and label (or other features) given in example.
    """
    if not calling:
        spec = {
            "image/encoded": tf.io.FixedLenFeature((), tf.string),
            "label": tf.io.FixedLenFeature((), tf.int64),
        }
    else:
        spec = {
            "image/encoded": tf.io.FixedLenFeature((), tf.string),
            "variant/encoded": tf.io.FixedLenFeature((), tf.string),
            "alt_allele_indices/encoded": tf.io.FixedLenFeature((), tf.string),
            "locus": tf.io.FixedLenFeature((), tf.string),
        }

    with tf.compat.v1.name_scope("input"):
        parsed = tf.io.parse_single_example(serialized=tf_example, features=spec)
        image = parsed["image/encoded"]
        image = tf.reshape(tf.io.decode_raw(image, tf.uint8), input_shape)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.subtract(image, 128.0)
        image = tf.math.divide(image, 128.0)

        if not calling:
            label = parsed["label"]
            label = tf.one_hot(label, n_classes, dtype=tf.float32)

            # if tuple a,b passed via --combine_labels, set b to 1 whenever a is 1
            if combine_labels:
                for cl in combine_labels:
                    if label[cl[0]] == 1:
                        label += tf.one_hot(cl[1], n_classes, dtype=tf.float32)
                        label *= 0.5

            return image, label
        else:
            return (
                image,
                parsed["variant/encoded"],
                parsed["alt_allele_indices/encoded"],
                parsed["locus"],
            )


def load_dataset(filename):
    """Loads an example dataset from tfrecord file."""
    dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
    return dataset


def create_dataset(
    file_list,
    batch_size,
    prefetch_size,
    input_shape,
    n_classes,
    shuffle_buffer=1000,
    training=True,
    calling=False,
    combine_labels=None,
):

    dataset = file_list.interleave(load_dataset)
    if training:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    dataset = dataset.map(
        map_func=(
            lambda x: parse_tfexample(
                x, input_shape, n_classes, calling, combine_labels
            )
        )
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(prefetch_size)
    return dataset


def get_pbtxt(sample_path):
    """Retrieves a pbtxt-file from sample_path, or creates one by counting examples if necessary.
    Args:
      sample_path: Path where pbtxt-file is looked up.
    Returns:
      Path to a pbtxt-file.
    """
    dataset_config_pbtxt = glob.glob(sample_path + "/*.pbtxt")
    if len(dataset_config_pbtxt) > 1:
        logging.warning(
            "More than one .pbtxt-files in directory {}; using first one: {}.".format(
                sample_path, dataset_config_pbtxt[0]
            )
        )
    elif len(dataset_config_pbtxt) == 0:
        logging.warning(
            "No .pbtxt-file found in {}. It is now created by counting entries; this may take a few minutes.".format(
                sample_path
            )
        )
        dataset_config_pbtxt = create_pbtxt(sample_path)

    return dataset_config_pbtxt[0]


def get_num_examples(sample_path):
    """Reads the total number of examples for one sample from a pbtxt file.
    Args:
      sample_path: Path where pbtxt-file is looked up.

    Returns:
      Number of examples for one sample.
    """
    dataset_config_pbtxt = get_pbtxt(sample_path)
    with open(dataset_config_pbtxt) as f:
        lines = f.read().splitlines()
    num_examples = [
        l.split(" ")[-1].replace('"', "") for l in lines if l.startswith("num_examples")
    ][0]
    return int(num_examples)


def get_tfrecord_pattern(sample_path):
    """Reads the patterns of tfrecord files for one sample from a pbtxt file.
    Args:
      sample_path: Path where pbtxt-file is looked up.

    Returns:
      Pattern of tfrecord files.
    """
    dataset_config_pbtxt = get_pbtxt(sample_path)
    with open(dataset_config_pbtxt) as f:
        lines = f.read().splitlines()
    tfrecord_pattern = [
        l.split(" ")[-1].replace('"', "")
        for l in lines
        if l.startswith("tfrecord_path")
    ][0]
    return tfrecord_pattern


def get_dataset(
    samples,
    batch_size,
    input_shape,
    n_classes,
    shuffle_buffer=1000,
    training=False,
    purpose="evaluation",
    calling=False,
    combine_labels=None,
):
    """Creates a tf.data.Dataset within a given directory containing subdirectories with tfexample files of one sample each.
    Args:
      samples: A list of paths of the samples to be incorporated in this dataset
      batch_size: Number of examples per batch in the dataset.
      input_shape: Shape of input tensors
      n_classes: Number of predicted classes
      shuffle_buffer: Window of tf.data.Dataset to consider for shuffling
      training: If true, datasets are repeated and shuffled
      purpose: A string to specify purpose of the dataset (training, validation, evaluation,...)

    Returns:
      A tf.data.Dataset
      The number of examples
    """

    families = set(["_".join(split("-|_", s.split("/")[-1])[0:2]) for s in samples])

    logging.info("\nFamilies in {} set:  {}\n\n".format(purpose, families))

    # read patterns of tfrecord files from .pbtxt-files in the directories
    tfrecord_patterns = list(map(get_tfrecord_pattern, samples))

    # expand the names of the tfrecord files
    tfrecords = []
    for p in tfrecord_patterns:
        tfrecords += glob.glob(p)
    if training and shuffle_buffer != 0:
        np.random.shuffle(tfrecords)

    if not tfrecords:
        logging.error("{} set is empty.".format(purpose))
        sys.exit(1)

    # create dataset
    dataset = create_dataset(
        tf.data.Dataset.list_files(tfrecords, shuffle=(shuffle_buffer != 0)),
        batch_size,
        tf.data.AUTOTUNE,
        input_shape,
        n_classes,
        shuffle_buffer,
        training=training,
        calling=calling,
        combine_labels=combine_labels,
    )

    # read number of examples from .pbtxt-files and sum them up
    nexamples = sum(list(map(get_num_examples, samples)))

    logging.info("\nLoaded {} examples for {}.\n\n".format(nexamples, purpose))

    return dataset, nexamples


def get_train_val_datasets(
    data_path,
    batch_size,
    input_shape,
    n_classes,
    shuffle_buffer=1000,
    training=True,
    val_split=0.2,
    combine_labels=None,
):
    """Creates a train/validation split for a given directory containing subdirectories with tfexample files of one sample each,
    and returns the corresponding tf.data.Datasets.
        Args:
          data_path: Path with a number of directories. These directories have to contain tfexample files for one sample and a dataset config pbtxt-file.
          batch_size: Number of examples per batch in training and validation datasets.
          input_shape: Shape of input tensors
          n_classes: Number of predicted classes
          shuffle_buffer: Window of tf.data.Dataset to consider for shuffling
          training: If true, datasets are repeated and shuffled
          val_split: Proportion of valiation samples

        Returns:
          A tf.data.Dataset for training
          The number of training examples
          A tf.data.Dataset for validation
          The number of validation examples
    """

    # list all subdirectories of data_path (this should be the directories with examples for the individual samples)
    samples = np.asarray(glob.glob(data_path + "/*"))
    # make sure that only directories are used for train/val split
    samples = samples[np.asarray(list(map(os.path.isdir, samples)))]

    # create train/val split
    np.random.seed(42)
    while True:
        # randomly select samples for validation set
        val_indices = np.random.choice(
            len(samples), int(np.floor(val_split * len(samples))), replace=False
        )
        train_samples = np.delete(np.array(samples), val_indices)
        val_samples = np.array(samples)[val_indices]

        # check if there is any overlap between train and val set w.r.t. families
        # TODO: Check if this is also applicable for other data (e.g., INOVA cohort)
        train_families, val_families = (
            set(["_".join(split("-|_", s.split("/")[-1])[0:2]) for s in train_samples]),
            set(["_".join(split("-|_", s.split("/")[-1])[0:2]) for s in val_samples]),
        )

        shared_families = train_families.intersection(val_families)

        # if there is an overlap: Continue
        if shared_families:
            continue
        # if not: Keep this train/val split
        else:
            break

    train_dataset, nexamples_train = get_dataset(
        train_samples,
        batch_size,
        input_shape,
        n_classes,
        shuffle_buffer,
        training=training,
        purpose="training",
        combine_labels=combine_labels,
    )

    val_dataset, nexamples_val = get_dataset(
        val_samples,
        batch_size,
        input_shape,
        n_classes,
        shuffle_buffer,
        training=training,
        purpose="validation",
        combine_labels=combine_labels,
    )

    return train_dataset, nexamples_train, val_dataset, nexamples_val


def get_cv_datasets(
    dataset_path,
    batch_size,
    input_shape,
    n_classes,
    shuffle_buffer=1000,
    training=True,
    n_folds=5,
    combine_labels=None,
):
    """Implements cross validation for a given directory containing subdirectories with tfexample files of one sample each.
    Args:
      data_path: Path with a number of directories. These directories have to contain tfexample files for one sample and a dataset config pbtxt-file.
      batch_size: Number of examples per batch in training and validation datasets.
      input_shape: Shape of input tensors
      n_classes: Number of predicted classes
      shuffle_buffer: Window of tf.data.Dataset to consider for shuffling
      training: If true, datasets are repeated and shuffled
      n_folds: Number of folds for cross validation.

    Returns:
      A generator object yielding a total of n_folds tuples (train_dataset, nexamples_train, val_dataset, nexamples_val).
    """

    # get list of samples from dataset_path directory
    samples = np.asarray(glob.glob(dataset_path + "/*"))
    # make sure that only directories are used
    samples = samples[np.asarray(list(map(os.path.isdir, samples)))]
    # map to only family ID, so that families will always be together in one fold (probably not applicable to other datasets)
    families = np.asarray(list(set([s[0:-7] for s in samples])))
    # shuffle families randomly
    np.random.shuffle(families)

    # determine the size of each subset to be used for validation
    val_split = np.ceil(len(families) / n_folds)

    # now create n_folds folds
    folds_arrays = np.array_split(families, len(families) // val_split + 1)
    folds = [list(fold) for fold in folds_arrays]

    for fold in folds:
        # get train and val samples from folds
        train_families = list(set(families) - set(fold))
        val_families = fold

        # expand from families to samples again
        train_samples = [
            y
            for x in list(map(glob.glob, [t + "*" for t in train_families]))
            for y in x
        ]
        val_samples = [
            y for x in list(map(glob.glob, [v + "*" for v in val_families])) for y in x
        ]

        # finally yield the two datasets (and their cardinalities) for this fold
        train_dataset, nexamples_train = get_dataset(
            train_samples,
            batch_size,
            input_shape,
            n_classes,
            shuffle_buffer,
            purpose="training",
            training=training,
            combine_labels=combine_labels,
        )

        val_dataset, nexamples_val = get_dataset(
            val_samples,
            batch_size,
            input_shape,
            n_classes,
            shuffle_buffer,
            purpose="validation",
            training=training,
            combine_labels=combine_labels,
        )

        yield train_dataset, nexamples_train, val_dataset, nexamples_val


def get_msdn_id(msdn_ranges, msdn_contig, msdn_position):
    """Updates list of MSDN ranges and returns position of a new MSDN in this list.
    Args:
        msdn_ranges: A list of MSDN boundaries (or empty list).
        msdn_contig: The contig of the new MSDN.
        msdn_position: The position of the new MSDN.
    """
    # check if new MSDN falls into the boundaries of one that has already been stored
    msdn_ranges_new = [
        (t[0], min(t[1], msdn_position - 20), max(t[2], msdn_position + 20))
        if ((t[0] == msdn_contig) and (t[1] <= msdn_position <= t[2]))
        else t
        for t in msdn_ranges
    ]
    # if the new list equals the old: New MSDN did not fall in existing boundaries and a new entry is added
    if msdn_ranges_new == msdn_ranges:
        msdn_ranges_new += [(msdn_contig, msdn_position - 20, msdn_position + 20)]
    # position of difference between msdn_ranges_new and msdn_ranges is the index of the MSDN cluster
    msdn_id = [i for i, el in enumerate(msdn_ranges_new) if el not in msdn_ranges][0]

    return msdn_id, msdn_ranges_new


def write_vcf(
    examples_batch,
    probs_batch,
    copied_header,
    output_vcf,
    dnm_threshold,
    msdn_threshold,
    class_labels,
    msdn_ranges,
):
    """Calls and metadata from a batch are written to a VCF file.
    Args:
      examples_batch: A batch of examples as read from tfrecord file.
      probs_batch: Class probabilities predicted for the batch by the model.
      copied_header: VCF header to which the new variants are appended.
      output_vcf: Path to the output VCF.
      dnm_threshold: Threshold of phred-scaled error probability above which a DNM call is flagged as DNM in VCF.
      class_labels: A dictionary assigning each class label to a number.
      msdn_ranges: A list of MSDN boundaries (or empty list).
    """

    loci_batch = examples_batch[-1].numpy()
    variants_encoded_batch = examples_batch[1].numpy()
    alt_alleles_encoded_batch = examples_batch[2].numpy()

    for locus, variant_encoded, alt_alleles_encoded, probs in zip(
        loci_batch, variants_encoded_batch, alt_alleles_encoded_batch, probs_batch
    ):
        locus = locus.decode("utf-8").split(":")
        contig = locus[0]
        start = int(locus[1].split("-")[0]) - 1
        stop = int(locus[1].split("-")[1])
        variant = variants_pb2.Variant.FromString(variant_encoded)
        alt_alleles = deepvariant_pb2.CallVariantsOutput.AltAlleleIndices.FromString(
            alt_alleles_encoded
        )

        call_variants_output = deepvariant_pb2.CallVariantsOutput(
            variant=variant, alt_allele_indices=alt_alleles
        )

        ref_allele = call_variants_output.variant.reference_bases
        alt_alleles = call_variants_output.variant.alternate_bases

        af = call_variants_output.variant.calls[0].info["VAF"].values[0].number_value
        dp = call_variants_output.variant.calls[0].info["DP"].values[0].int_value

        new_record = copied_header.new_record(
            contig=contig,
            start=start,
            stop=stop,
            alleles=(ref_allele, ",".join(alt_alleles)),
            info={"DP": dp, "AF": af},
        )

        # combine all de-novo classes with HET
        if "DNM" in class_labels.keys():
            p_denovo_phred = _phred_score(1 - (probs[class_labels["DNM"]].item()))

            probs[class_labels["HET"]] += probs[[class_labels["DNM"]]]
            if "MSDN" in class_labels.keys():
                # p_denovo_phred = -10*math.log10(1-(probs[class_labels["DNM"]] + probs[class_labels["MSDN"]].item()))
                p_msdn_phred = _phred_score(
                    1 - (probs[class_labels["MSDN"]].item())
                )
                # probs[class_labels["DNM"]] += probs[[class_labels["MSDN"]]]
                probs[class_labels["HET"]] += probs[[class_labels["MSDN"]]]

        # keep only three base classes
        probs_gt = probs[
            [class_labels["HOM"], class_labels["HET"], class_labels["ALT"]]
        ]

        probs_gt = list(map(lambda x: int(_phred_score(x)), probs_gt))
        genotype = np.argmin(probs_gt)
        if genotype == 0:
            new_record.samples[copied_header.samples[0]]["GT"] = (0, 0)
        elif genotype == 1:
            new_record.samples[copied_header.samples[0]]["GT"] = (0, 1)
            if "DNM" in class_labels.keys():
                new_record.samples[copied_header.samples[0]]["DQ"] = p_denovo_phred
                if p_denovo_phred >= dnm_threshold:
                    new_record.samples[copied_header.samples[0]]["DN"] = "DeNovo"
                else:
                    new_record.samples[copied_header.samples[0]]["DN"] = "LowDQ"
                if "MSDN" in class_labels.keys():
                    if p_msdn_phred >= msdn_threshold:
                        # add p_DNM and p_MSDN up for MSDQ
                        new_record.samples[copied_header.samples[0]][
                            "MSDQ"
                        ] = _phred_score(
                            1
                            - (
                                probs[class_labels["DNM"]]
                                + probs[class_labels["MSDN"]].item()
                            )
                        )
                        new_record.samples[copied_header.samples[0]][
                            "MSDN"
                        ] = "MultisiteDeNovo"
                        # MSDN are also DNM
                        new_record.samples[copied_header.samples[0]]["DN"] = "DeNovo"
                        # print(
                        #     probs[class_labels["DNM"]]
                        #     + probs[class_labels["MSDN"]].item()
                        # )
                        # get id of the MSDN cluster
                        msdn_id, msdn_ranges = get_msdn_id(
                            msdn_ranges=msdn_ranges,
                            msdn_contig=locus[0],
                            msdn_position=int(locus[1].split("-")[1]),
                        )
                        new_record.samples[copied_header.samples[0]]["MSDID"] = msdn_id
                    else:
                        new_record.samples[copied_header.samples[0]]["MSDN"] = "LowMSDQ"
        elif genotype == 2:
            new_record.samples[copied_header.samples[0]]["GT"] = (1, 1)

        new_record.samples[copied_header.samples[0]]["PL"] = tuple(probs_gt)

        output_vcf.write(new_record)

    return msdn_ranges


# msdn_ranges = write_vcf(examples, net_output, copied_header, output_vcf, dnm_threshold_phred, msdn_threshold_phred, class_labels, msdn_ranges)
def to_table_row(
    examples,
    net_output,
    dnm_threshold=0.90,
    msdn_threshold=0.90,
    dnm_class=3,
    msdn_class=4
):
    dnm_indices = None
    if msdn_class is not None:
        dnm_indices = np.where(
            (net_output[:, dnm_class] > dnm_threshold)
            | (net_output[:, msdn_class] > msdn_threshold)
        )[0].tolist()
    else:
        dnm_indices = np.where(
            (net_output[:, dnm_class] > dnm_threshold)
        )[0].tolist()

    dnm_loci_batch = examples[-1].numpy()[dnm_indices].tolist()
    dnm_loci_batch = [[locus.decode("utf-8").split("-")[0]] for locus in dnm_loci_batch]

    variants_batch = [
        variants_pb2.Variant.FromString(encoded_variant)
        for encoded_variant in examples[1].numpy()[dnm_indices]
    ]
    alt_allele_indices_batch = [
        deepvariant_pb2.CallVariantsOutput.AltAlleleIndices.FromString(encoded_aa)
        for encoded_aa in examples[2].numpy()[dnm_indices]
    ]

    ref_alleles_batch = [None] * len(dnm_indices)
    alt_alleles_batch = [None] * len(dnm_indices)
    VAF_batch = [None] * len(dnm_indices)
    DP_batch = [None] * len(dnm_indices)

    for i in range(len(dnm_indices)):
        call_variants_output = deepvariant_pb2.CallVariantsOutput(
            variant=variants_batch[i], alt_allele_indices=alt_allele_indices_batch[i]
        )
        # genotype_probabilities = list(net_output)[0])
        ref_alleles_batch[i] = call_variants_output.variant.reference_bases
        alt_alleles_batch[i] = call_variants_output.variant.alternate_bases
        VAF_batch[i] = (
            call_variants_output.variant.calls[0].info["VAF"].values[0].number_value
        )
        DP_batch[i] = (
            call_variants_output.variant.calls[0].info["DP"].values[0].int_value
        )

    probabilities_batch = net_output[dnm_indices]

    # filtering for SNPs and dnm_threshold here
    # output = [(locus[0], prop[dnm_class], ref, alt, vaf, dp) for locus, prop, ref, alt, vaf, dp in zip(dnm_loci_batch, probabilities_batch, ref_alleles_batch,alt_alleles_batch, VAF_batch, DP_batch) if (prop[dnm_class] >= dnm_threshold) and len(ref)==1 and len(alt)==1 and len(alt[0])==1]
    output = [
        (locus[0], prop[dnm_class], prop[msdn_class] if msdn_class is not None else np.nan, ref, alt, vaf, dp)
        for locus, prop, ref, alt, vaf, dp in zip(
            dnm_loci_batch,
            probabilities_batch,
            ref_alleles_batch,
            alt_alleles_batch,
            VAF_batch,
            DP_batch,
        )
        if ((prop[dnm_class] >= dnm_threshold) or ((prop[msdn_class] if msdn_class is not None else np.nan) >= msdn_threshold))
        and len(ref) == 1
        and len(alt) == 1
        and len(alt[0]) == 1
    ]
    # output = [(locus[0], prop[args.dnm_class]) for locus, prop in zip(dnm_loci_batch, probabilities_batch) if (prop[args.dnm_class] >= args.dnm_threshold)]
    return output
