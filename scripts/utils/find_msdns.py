# by Fabian Brand, from https://git.meb.uni-bonn.de/brand/msdns-hail.git

#!/usr/bin/env python
# coding: utf-8

import pysam
import hail as hl
import hail.expr.aggregators as agg
from bokeh.io import output_notebook, show
import pandas
import sys
import os
import argparse
import logging
import functools

##
# Logging
logger = logging.getLogger("find-msdns")
logger.setLevel(logging.ERROR)
channel = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s - %(name)s - %(levelname)7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
channel.setFormatter(formatter)
logger.addHandler(channel)

log_level = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}

##
# Checkpoints
#
checkpoints = set([
    'de_novo',
    'refined_dnm',
    'msdns',
    'filter_msdns',
])
up_to_check = {
    'de_novo': 1,
    'refined_dnm': 2,
    'msdns': 3,
    'filter_msdns': 4,
}

##
# Parsing Arguments and sanitizing
#

parser = argparse.ArgumentParser(
    prog="FIND-MSDNS",
    description='Find msdns in a given vcf file.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('vcf', metavar='VCF', type=str,
                    help='VCF input file (prefer bgziped files)')
parser.add_argument('-p', '--output', type=str, default='output',
                    help='Output file prefix directory')
parser.add_argument('-f', '--fam', type=str, required=True,
                    help='Pedigree file (plink .fam format)')
parser.add_argument('-R', '--reference', type=str, required=True,
                    help='Reference sequence')
parser.add_argument('-U', '--up-to', choices=checkpoints, default='filter_msdns',
                    help='Process up to named checkpoint')
parser.add_argument('--min-p-de-novo', type=float, default=0.80,
                    help='Min. probability of de novo event')
parser.add_argument('--min-aaf', type=float, default=0.30,
                    help='Min. alternate allele frequency to call het')
parser.add_argument('--min-parent-dp', type=int, default=10,
                    help='Min. sequencing depth in parents')
parser.add_argument('--min-dp', type=int, default=15,
                    help='Min. sequencing depth in index')
parser.add_argument('--max-parent-alleles', type=int, default=1,
                    help='Max. reads in parents supporting alt allele')
parser.add_argument('--window-size', type=int, default=20,
                    help='MSDN window size')
parser.add_argument('-t', '--tmp-dir', help='Temporary directory',
                    type=str, default=os.getenv('TMPDIR', '/tmp'))
parser.add_argument('--verbose', '-v', help='Set verbosity',
                    choices=log_level.keys(), default='info')

args = parser.parse_args()
logger.setLevel(log_level[args.verbose])
logger.debug("Arguments: {}".format(vars(args)))

if not os.path.isfile(args.reference) or not os.path.isfile(args.reference + ".fai"):
    logger.error("Could not find reference file and index at {}".format(args.reference))
    sys.exit(1)

if not os.path.isfile(args.vcf):
    logger.error("Could not find vcf file at {}".format(args.vcf))
    sys.exit(1)

hl.init(
    idempotent=True,
    tmp_dir=args.tmp_dir,
    log=os.path.join(args.tmp_dir, 'hail.log'),
    # spark_conf={
    #     "spark.local.dir": args.tmp_dir,
    # }
)

##
# Functions and wrapper
#
def checkpoint(*, prefix=args.output, name=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            file_path = os.path.join(prefix, name)
            if name is None:
                return fn(*args, **kwargs)
            else:
                if os.path.exists(file_path):
                    logger.info("Resuming from checkpoint: {}".format(file_path))
                    if file_path.endswith('mt'):
                        logger.info("Returning checkpoint table")
                        return hl.read_matrix_table(file_path)
                    elif file_path.endswith('ht'):
                        return hl.read_table(file_path)
                    else:
                        logger.error("Unknown file format for checkpoint: {}".format(file_path))
                        sys.exit(1)
                else:
                    logger.info("Generating data for checkpoint: {}".format(file_path))
                    table = fn(*args, **kwargs)
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    if file_path.endswith('mt'):
                        table.write(file_path, overwrite=True)
                    elif file_path.endswith('ht'):
                        table.write(file_path, overwrite=True)
                    else:
                        logger.error("Unknown file format for checkpoint: {}".format(file_path))
                        sys.exit(1)
                    return table
        return wrapper
    return decorator

@checkpoint(name='r_de_novo_mt.mt')
def find_de_novo(data, pedigree, de_novo_threshold=0.7):
    # The following line is not supported in hail 0.2.11 (but later versions)
    # data = hl.split_multi_hts(data, permit_shuffle=True)
    data = hl.split_multi_hts(data)
    data = data.annotate_rows(
        AC=data.info.AC[data.a_index - 1],
        iAF=data.info.AF[data.a_index - 1]
    )
    data = hl.sample_qc(data)
    data = hl.variant_qc(data)

    print("Applying de novo filter...")
    radar_ped = hl.Pedigree.read(pedigree)
    de_novo_scores = hl.de_novo(data, radar_ped, pop_frequency_prior=data.variant_qc.AF[-1], min_gq=0)# for deeptrio
    de_novo_mt = de_novo_scores.to_matrix_table(row_key=['locus', 'alleles'], col_key=['id'])

    de_novo_data = data.annotate_entries(p_de_novo=de_novo_mt[(data.locus, data.alleles), data.s].p_de_novo)

    print("Annotating trio data...")
    trio_mt = hl.trio_matrix(de_novo_data, radar_ped, complete_trios=True)

    de_novo_data = de_novo_data.annotate_entries(
        mother=trio_mt[(de_novo_data.locus, de_novo_data.alleles), de_novo_data.s].mother_entry,
        father=trio_mt[(de_novo_data.locus, de_novo_data.alleles), de_novo_data.s].father_entry,
    )

    de_novo_data = de_novo_data.filter_entries(hl.is_defined(de_novo_data.GT)
        & de_novo_data.GT.is_non_ref()
        & (de_novo_data.p_de_novo > de_novo_threshold)
    )

    r_de_novo_mt = de_novo_data.select_cols()
    r_de_novo_mt = r_de_novo_mt.select_rows('AC', 'iAF')
    r_de_novo_mt = r_de_novo_mt.select_entries('AD', 'DP', 'GT', 'p_de_novo',
        mother=hl.Struct(**{
            'AD': r_de_novo_mt.mother.AD,
            'AF': r_de_novo_mt.mother.AD[1] / hl.sum(r_de_novo_mt.mother.AD),
            'DP': r_de_novo_mt.mother.DP,
            'GT': r_de_novo_mt.mother.GT
        }),
        father=hl.Struct(**{
            'AD': r_de_novo_mt.father.AD,
            'AF': r_de_novo_mt.father.AD[1] / hl.sum(r_de_novo_mt.father.AD),
            'DP': r_de_novo_mt.father.DP,
            'GT': r_de_novo_mt.father.GT        
        })
    )
    r_de_novo_mt = r_de_novo_mt.annotate_entries(AF=r_de_novo_mt.AD[1] / hl.sum(r_de_novo_mt.AD))
    return r_de_novo_mt

@checkpoint(name='dnm.ht')
def refine_dnm(data, min_p_de_novo=args.min_p_de_novo, min_aaf=args.min_aaf,
               max_parent_alleles=args.max_parent_alleles,
               min_dp=args.min_dp, min_parent_dp=args.min_parent_dp):
    dnm = data.filter_entries(
        (data.p_de_novo > min_p_de_novo)
        & (data.DP >= min_dp)
        & (data.mother.DP >= min_parent_dp)
        & (data.father.DP >= min_parent_dp)
        & (data.AF > min_aaf)
        & (data.AF < 1 - min_aaf)
    )
    dnm_ht = dnm.key_cols_by().entries()
    dnm_ht = dnm_ht.filter(
        (dnm_ht.alleles[0].length()==1)
        & (dnm_ht.alleles[1].length()==1)
    )
    dnm_ht = dnm_ht.filter(
        (dnm_ht.father.AD[1] <= max_parent_alleles)
        & (dnm_ht.mother.AD[1] <= max_parent_alleles)
    )
    return dnm_ht

@checkpoint(name='clustered_dnm.mt')
def clustered_dnm(data, window_size=args.window_size):
    # Use hail 0.2.11 for this part, the method was removed in later versions
    clustered_dnm = hl.methods.window_by_locus(data, window_size)
    clustered_dnm = clustered_dnm.select_cols()
    clustered_dnm = clustered_dnm.filter_entries(
        (hl.is_defined(clustered_dnm.GT)
        & (clustered_dnm.prev_entries.length() > 0))
    )
    clustered_dnm = clustered_dnm.filter_entries(
        clustered_dnm.prev_entries.filter(lambda x: hl.is_defined(x.GT) & x.GT.is_non_ref()).length() > 0
    )
    return clustered_dnm

@checkpoint(name='msdns.ht')
def filter_msdns(data, max_parent_alleles=args.max_parent_alleles):
    # Explode and annotate dist
    msdn_ht = data.key_cols_by().entries()
    msdn_ht = msdn_ht.annotate(indices = hl.range(0, hl.len(msdn_ht.prev_rows)))
    msdn_ht = msdn_ht.explode('indices')
    msdn_ht = msdn_ht.annotate(
        previous=hl.Struct(**{
            **msdn_ht.prev_rows[msdn_ht.indices],
            **msdn_ht.prev_entries[msdn_ht.indices]
        })
    )
    msdn_ht = msdn_ht.filter(
        hl.is_defined(msdn_ht.previous.GT)
        & msdn_ht.previous.GT.is_non_ref()
    )
    msdn_ht = msdn_ht.annotate(
        dist=msdn_ht.locus.position - msdn_ht.previous.locus.position
    )

    # Filter SNPs
    msdn_ht = msdn_ht.filter(msdn_ht.dist > 0)
    msdn_ht = msdn_ht.filter((msdn_ht.alleles[0].length()==1) & (msdn_ht.alleles[1].length()==1)
        & (msdn_ht.previous.alleles[0].length()==1) & (msdn_ht.previous.alleles[1].length()==1))

    # Remove artifacts with substantial reads in parents
    par_alleles_excl = max_parent_alleles + 1
    msdn_ht = msdn_ht.filter(
        ((msdn_ht.father.AD[1] < par_alleles_excl)
            | (msdn_ht.previous.father.AD[1] < par_alleles_excl))
        & ((msdn_ht.mother.AD[1] < par_alleles_excl) 
            | (msdn_ht.previous.mother.AD[1] < par_alleles_excl))
    )

    msdn_ht = msdn_ht.select(
        's', 'AC', 'iAF', 'AD', 'AF', 'DP',
        'GT', 'p_de_novo', 'dist', 'previous', 'mother', 'father'
    )
    return msdn_ht

##
# Script
#
rg = hl.get_reference('GRCh37')
rg.add_sequence(args.reference, args.reference + ".fai")
logger.info("Importing vcf file...")
data = hl.import_vcf(
    args.vcf, call_fields=["GT"], skip_invalid_loci=True,
    array_elements_required=False, find_replace=("-inf", "-100")
)

r_de_novo_mt = find_de_novo(data, args.fam)
if args.up_to:
    if up_to_check[args.up_to] > 1:
        dnm_ht = refine_dnm(r_de_novo_mt)
        dnm_pandas = dnm_ht.to_pandas()
        dnm_pandas.to_csv(os.path.join(args.output, "dnm.tsv"), index=False, sep="\t")
    if up_to_check[args.up_to] > 2:
        msdns = clustered_dnm(r_de_novo_mt)
    if up_to_check[args.up_to] > 3:
        msdn_ht = filter_msdns(msdns)
        logger.info("Exporting MSDN tables to pandas and .tsv file")
        df = msdn_ht.to_pandas()
        df.to_csv(os.path.join(args.output, "msdns.tsv"), index=False, sep="\t")
else:
    logger.info("Ignoring msdn detection and filtration due to 'up-to' setting")
