from absl import logging
import argparse
import config
import os
import glob
import numpy as np
import pysam
import pandas
import math
from time import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # to silence excessive TF output
import tensorflow as tf
from utils.model_utils import load_weights_layerwise
from utils.io_utils import get_dataset, write_vcf, to_table_row

log_level = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

parser = argparse.ArgumentParser(
    prog="CALL DE-NOVO MUTATIONS",
    description="Use a trained model to call de-novo mutations in given examples.",
)

# required
parser.add_argument(
    "--call_data_path",
    type=str,
    required=True,
    help="The path of the pile-up images to call de-novo mutations.",
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="The path of the model checkpoint to be be used for calling.",
)
parser.add_argument(
    "--output_vcf",
    type=str,
    required=True,
    help="The path where vcf with de-novo calls will be written.",
)
parser.add_argument(
    "--copy_header_from",
    type=str,
    required=True,
    help="Path of a VCF file to provide header for the output VCF.",
)
parser.add_argument(
    "--output_table", type=str, required=True, help="Path to write the output table to"
)

# fine to run with default
parser.add_argument(
    "--input_shape",
    type=tuple,
    default=config.network_shape["input_shape"],
    help="The expected shape of input tensors (h,w,c).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=config.train_params["batch_size"],
    help="The number of samples in each batch.",
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=str,
    default=["HOM", "HET", "ALT", "DNM"],
    help='List of expected classes ("HOM", "HET", "ALT", "DNM", "MSDN", ...) ("MSDN" is ignored if "DNM" not given!)',
)
parser.add_argument(
    "--log_level",
    type=str,
    default=config.general_config["log_level"],
    help='Verbosity level ("error", "warning", "info", "debug").',
)
parser.add_argument(
    "--variable_path",
    type=str,
    default=config.general_config["variable_path"],
    help="The path of a .txt-file containing model variable names to be loaded into the model (needed if checkpoint is name-based as the ones that come with DeepVariant/DeepTrio).",
)
parser.add_argument(
    "--dnm_threshold",
    type=float,
    default=0.9,
    help="Threshold of probability to use to decide whether a variant is actually de novo.",
)
parser.add_argument(
    "--msdn_threshold",
    type=float,
    default=0.9,
    help="Threshold of probability to use to decide whether a variant is actually an MSDN.",
)

args = parser.parse_args()


def main():

    # list all subdirectories of eval_data_path (this should be the directories with examples for the individual samples)
    calling_samples = np.asarray(glob.glob(args.call_data_path + "/*"))
    # make sure that only directories are used
    calling_samples = calling_samples[
        np.asarray(list(map(os.path.isdir, calling_samples)))
    ]

    # create dataset from given config .pbtxt-file
    dataset_calling, nexamples_calling = get_dataset(
        calling_samples,
        args.batch_size,
        args.input_shape,
        len(args.classes),
        purpose="calling",
        training=False,
        calling=True,
    )

    try:
        # load model from file
        model = tf.keras.models.load_model(args.model_path)

    except:
        logging.warning(
            "Loading model weights with tf.keras.models.load_model(..) failed. Restoring weights layerwise instead."
        )

        # load basic inception_v3 model
        model = tf.keras.applications.InceptionV3(
            include_top=True,
            weights=None,
            input_shape=args.input_shape,
            classes=len(args.classes),
        )
        load_weights_layerwise(
            model,
            args.model_path,
            args.variable_path,
            output_layer=True,
            eval_mode=True,
        )

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

    logging.info("Starting calling loop.")

    class_labels = {}
    for c in ["HOM", "HET", "ALT", "DNM", "MSDN"]:
        if c in args.classes:
            class_labels[c] = args.classes.index(c)

    dnm_threshold_phred = -10 * math.log10(1 - args.dnm_threshold)
    msdn_threshold_phred = -10 * math.log10(1 - args.msdn_threshold)

    copied_header = pysam.VariantFile(args.copy_header_from).header.copy()

    # add a metadata line for DN containing information about DNM threshold (and dropping old one from copied header)
    for format_field in ["DQ", "DN", "MSDQ", "MSDN"]:
        if format_field in copied_header.formats.keys():
            copied_header.formats.remove_header(format_field)
    if not "DP" in copied_header.info.keys():
        copied_header.add_meta(
            "INFO",
            items=(
                ("ID", "DP"),
                ("Number", "1"),
                ("Type", "Integer"),
                ("Description", ""),
            ),
        )
    if not "AF" in copied_header.info.keys():
        copied_header.add_meta(
            "INFO",
            items=(
                ("ID", "AF"),
                ("Number", "."),
                ("Type", "Float"),
                ("Description", ""),
            ),
        )
    if not "PL" in copied_header.formats.keys():
        copied_header.add_meta(
            "FORMAT",
            items=(
                ("ID", "PL"),
                ("Number", "."),
                ("Type", "Integer"),
                ("Description", ""),
            ),
        )
    if "DNM" in args.classes:
        copied_header.add_meta(
            "FORMAT",
            items=(
                ("ID", "DQ"),
                ("Number", "1"),
                ("Type", "Float"),
                ("Description", "Phred-scaled de-novo likelihood."),
            ),
        )
        copied_header.add_meta(
            "FORMAT",
            items=(
                ("ID", "DN"),
                ("Number", "1"),
                ("Type", "String"),
                (
                    "Description",
                    "Possible values are 'DeNovo' or 'LowDQ'. Threshold for a passing de novo call is DQ >= {}".format(
                        round(dnm_threshold_phred, 3)
                    ),
                ),
            ),
        )
        if "MSDN" in args.classes:
            copied_header.add_meta(
                "FORMAT",
                items=(
                    ("ID", "MSDQ"),
                    ("Number", "1"),
                    ("Type", "Float"),
                    ("Description", "Phred-scaled multi-side de-novo likelihood."),
                ),
            )
            copied_header.add_meta(
                "FORMAT",
                items=(
                    ("ID", "MSDN"),
                    ("Number", "1"),
                    ("Type", "String"),
                    (
                        "Description",
                        "Possible values are 'MultisideDeNovo' or 'LowMSDQ'. Threshold for a passing de novo call is MSDQ >= {}".format(
                            round(msdn_threshold_phred, 3)
                        ),
                    ),
                ),
            )
            copied_header.add_meta(
                "FORMAT",
                items=(
                    ("ID", "MSDID"),
                    ("Number", "1"),
                    ("Type", "Integer"),
                    ("Description", "ID of the MSDN cluster that this DNM belongs to."),
                ),
            )

    output_vcf = pysam.VariantFile(args.output_vcf, "w", header=copied_header)

    # can this get any more efficient?
    msdn_ranges = []  # global list that will be filled with the boundaries of MSDNs
    table_data = []
    batch_no = 0
    start_time = time()
    for examples in dataset_calling:
        # forward pass through model
        net_output = model.predict(examples[0])
        # write batch to VCF
        msdn_ranges = write_vcf(
            examples,
            net_output,
            copied_header,
            output_vcf,
            dnm_threshold_phred,
            msdn_threshold_phred,
            class_labels,
            msdn_ranges,
        )
        table_data.extend(
            to_table_row(
                examples,
                net_output,
                dnm_threshold=args.dnm_threshold,
                msdn_threshold=args.msdn_threshold,
                dnm_class=class_labels["DNM"],
                msdn_class=class_labels["MSDN"] if "MSDN" in class_labels else None,
            )
        )
        batch_no += 1
        if batch_no % 25 == 0:
            logging.info(f"processed {((batch_no * args.batch_size) / nexamples_calling) * 100:.02f}% ({batch_no / (time() - start_time):.02f} batches/s)")

    if not os.path.exists(os.path.dirname(args.output_table)):
        os.makedirs(os.path.dirname(args.output_table), exist_ok=True)
    out_df = pandas.DataFrame(
        table_data,
        columns=[
            "Locus",
            "Probability_DNM",
            "Probability MSDN",
            "Ref_allele",
            "Alt_alleles",
            "VAF",
            "DP",
        ],
    )
    out_df.to_csv(args.output_table, sep="\t", index=False)


if __name__ == "__main__":
    logging.set_verbosity(log_level[args.log_level])

    main()
