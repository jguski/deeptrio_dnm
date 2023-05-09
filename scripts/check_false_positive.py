import os
import argparse
import logging
import pysam
import pandas
from typing import Optional


log_level = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    prog="CHECK FALSE POSITIVE",
    description="Find likely false positive variants from a deep trio model for further fine tuning on these examples.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--truth_vcf",
    type=str,
    required=True,
    help="Initial truth vcf the examples were generated from",
)
parser.add_argument(
    "--call_table",
    type=str,
    required=True,
    help=".tsv table containing variant calls from a deep trio model",
)
parser.add_argument(
    "--trio_vcf",
    type=str,
    default=None,
    help="Family VCF to search for paternal allele frequencies in. If not specified, no revised calls based on parental information can be generated",
)
parser.add_argument(
    "--sample",
    type=str,
    required=True,
    help="Sample name for which the calls are evaluated",
)

# Optionals
parser.add_argument(
    "--dnm_threshold",
    type=float,
    default=0.9,
    help="Threshold to further filter the call table for high confidence de novo variants",
)
parser.add_argument(
    "--min_aaf",
    type=float,
    default=0.3,
    help="Minimum alternate allele frequency to consider a heterozygous call to be true positive",
)
parser.add_argument(
    "--max_aaf_parents",
    type=float,
    default=0.1,
    help="Maximum tolerable allele frequency in parents, before a heterozygous call is revised as inherited instead of de novo",
)
parser.add_argument(
    "--locus_col",
    type=str,
    default="Locus",
    help="Locus column header in the call table",
)
parser.add_argument(
    "--dnm_probability_col",
    type=str,
    default="Probability_DNM",
    help="Column header in the call table that contains the dnm probabilities",
)
parser.add_argument(
    "--vaf_col",
    type=str,
    default="VAF",
    help="Column name of the column containing variant allele frequencies in the call table",
)
parser.add_argument(
    "--dp_col",
    type=str,
    default="DP",
    help="Column name of the column containing depth information in the call table",
)
parser.add_argument(
    "--ref_col",
    type=str,
    default="Ref_allele",
    help="Column name for the column containing the reference allele in the call table",
)
parser.add_argument(
    "--alt_col",
    type=str,
    default="Alt_alleles",
    help="Column name for the column containing alternate alleles in the call table",
)
parser.add_argument(
    "--verbose", choices=log_level.keys(), default="info", help="Set output verbosity"
)


def get_variant_calls(
    table: str,
    dnm_threshold: float = 0.9,
    locus_col: str = "Locus",
    dnm_probability_col: str = "Probability_DNM",
) -> pandas.DataFrame:
    callset = pandas.read_csv(table, sep="\t")
    callset = callset[callset[locus_col].str.split(":").str[0] != "X"]
    callset = callset[callset[dnm_probability_col] > dnm_threshold]
    logger.info("read {} variant calls".format(len(callset)))
    return callset


def get_low_aaf_records(
    callset: pandas.DataFrame,
    min_aaf: float = 0.3,
    vaf_col: str = "VAF",
    locus_col: str = "Locus",
    dp_col: str = "DP",
    ref_col: str = "Ref_allele",
    alt_col: str = "Alt_alleles",
):
    for _, row in callset[callset[vaf_col] < min_aaf].iterrows():
        contig, pos = row[locus_col].split(":")
        logger.debug("got de novo with low aaf at: {}:{}".format(contig, pos))
        yield {
            "contig": contig,
            "start": int(pos) - 1,
            "stop": int(pos),
            "info": {"label": "0", "DP": row[dp_col], "AF": row[vaf_col]},
            "alleles": (row[ref_col], row[alt_col][2]),
            "filter": "PASS",
        }


def get_inherited_records(
    callset: pandas.DataFrame,
    trio_vcf: str,
    sample: str,
    min_aaf: float = 0.3,
    max_aaf_parents: float = 0.1,
    vaf_col: str = "VAF",
    locus_col: str = "Locus",
    dp_col: str = "DP",
    ref_col: str = "Ref_allele",
    alt_col: str = "Alt_alleles",
):
    in_vcf = pysam.VariantFile(trio_vcf, "r")
    try:
        candidates = callset[callset[vaf_col] >= min_aaf]
        logger.info("checking {} variants for inherited sites".format(len(candidates)))
        for _, row in candidates.iterrows():
            contig, pos = row[locus_col].split(":")
            for record in in_vcf.fetch(
                contig=contig, start=int(pos) - 1, stop=int(pos)
            ):
                try:
                    af_others = [
                        record.samples[s]["AF"][0]
                        for s in list(record.samples)
                        if s != sample
                    ]
                    if (
                        any(x > max_aaf_parents for x in af_others)
                        and len(record.alts) == 1
                    ):
                        logger.debug(
                            "got likely inherited position: {}:{}".format(contig, pos)
                        )
                        yield {
                            "contig": contig,
                            "start": int(pos) - 1,
                            "stop": int(pos),
                            "info": {
                                "label": "1",
                                "DP": row[dp_col],
                                "AF": row[vaf_col],
                            },
                            "alleles": (row[ref_col], row[alt_col][2]),
                            "filter": "PASS",
                        }
                except Exception as err:
                    logger.warning(
                        "got error processing site {}:{}: {}".format(contig, pos, err)
                    )
    finally:
        in_vcf.close()


def process(
    truth_vcf: str,
    call_table: str,
    sample: str,
    trio_vcf: Optional[str] = None,
    dnm_threshold: float = 0.9,
    dnm_probability_col: str = "Probability_DNM",
    min_aaf: float = 0.3,
    max_aaf_parents: float = 0.1,
    vaf_col: str = "VAF",
    locus_col: str = "Locus",
    dp_col: str = "DP",
    ref_col: str = "Ref_allele",
    alt_col: str = "Alt_alleles",
):
    truth_vcf_in = pysam.VariantFile(truth_vcf, "r")
    header = truth_vcf_in.header.copy()
    truth_vcf_in.close()

    output_vcf = pysam.VariantFile("-", "w", header=header)

    callset = get_variant_calls(
        call_table,
        dnm_threshold=dnm_threshold,
        locus_col=locus_col,
        dnm_probability_col=dnm_probability_col,
    )
    for record_data in get_low_aaf_records(
        callset,
        min_aaf=min_aaf,
        vaf_col=vaf_col,
        locus_col=locus_col,
        dp_col=dp_col,
        ref_col=ref_col,
        alt_col=alt_col,
    ):
        rec = header.new_record(**record_data)
        header.samples[sample]["GT"] = (0, 0)
        output_vcf.write(rec)

    if trio_vcf is not None and os.path.isfile(trio_vcf):
        for record_data in get_inherited_records(
            callset,
            trio_vcf,
            sample,
            min_aaf=min_aaf,
            max_aaf_parents=max_aaf_parents,
            vaf_col=vaf_col,
            locus_col=locus_col,
            dp_col=dp_col,
            ref_col=ref_col,
            alt_col=alt_col,
        ):
            rec = header.new_record(**record_data)
            header.samples[sample]["GT"] = (0, 1)
            output_vcf.write(rec)
    else:
        logger.warning("no trio vcf supplied, or the file does not exist")


def main():
    args = parser.parse_args()
    logger.setLevel(log_level[args.verbose])
    logger.debug("Arguments: {}".format(vars(args)))

    process(
        args.truth_vcf,
        args.call_table,
        args.sample,
        trio_vcf=args.trio_vcf,
        dnm_threshold=args.dnm_threshold,
        dnm_probability_col=args.dnm_probability_col,
        min_aaf=args.min_aaf,
        max_aaf_parents=args.max_aaf_parents,
        vaf_col=args.vaf_col,
        locus_col=args.locus_col,
        dp_col=args.dp_col,
        ref_col=args.ref_col,
        alt_col=args.alt_col,
    )


if __name__ == "__main__":
    main()
