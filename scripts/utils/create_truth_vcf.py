import argparse
import sys
import os
from logging import getLogger, DEBUG
import pandas as pd
import hail as hl
import numpy as np


logger = getLogger()
logger.setLevel(DEBUG)

parser = argparse.ArgumentParser(
    prog="CREATE TRUTH VCF",
    description='Short script to create truth VCF for retraining. Either 3 (HOM, HET, ALT), 4 (HOM, HET, ALT, DNM) or 5 (HOM, HET, ALT, DNM, MSDN) classes.')

parser.add_argument('--index_id', type=str, help='The ID of the genome for which the truth VCF file is being created.')
parser.add_argument('--vcf', type=str, help='VCF file (bgzipped) with variant calls to be annotated with class labels.')
parser.add_argument('--dnm1', type=str, default=None, help='.tsv-file with a list of filtered de-novo variants from one caller (e.g., DRAGEN).')
parser.add_argument('--dnm2', type=str, default=None, help='.tsv-file with a list of filtered de-novo variants from another caller (e.g., DeepVariant).')
parser.add_argument('--msdn1', type=str, default=None, help='.tsv-file with a list of filtered MSDN variants from one caller (e.g., DRAGEN).')
parser.add_argument('--msdn2', type=str, default=None, help='.tsv-file with a list of filtered MSDN variants from another caller (e.g., DeepVariant).')
parser.add_argument('--out_dir', type=str, help='The path the resulting truth VCF is written to.')
parser.add_argument('--tmp_dir', type=str, help='Directory for temporary files.')
parser.add_argument('--contigs', type=float, nargs="*", default=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"], help='Contigs to include in truth VCF.')
parser.add_argument('--prop_std_classes', type=float, default=1.0, help='The proportion of each standard class (HOM, HET, ALT) frequency to frequency of class DNM.')
parser.add_argument('--indel_window_file', type=str, default=None, help='Window in which to look for INDELS. If a potential INDEL is found within the window around a DNM candidate (that lies outside the overlap of the two callers), this DNM candidate will be labelled as HOM. Will be ignored if --dnm1 or --dnm2 not passed.')
parser.add_argument('--first_round_file', type=str, default=None, help='A separate VCF file that contains prefiltered sites that have been called before and that are probably artifacts.')

args = parser.parse_args()
logger.debug("Arguments: {}".format(vars(args)))

hl.init(idempotent=True,
    tmp_dir=args.tmp_dir,
    log=os.path.join(args.tmp_dir, 'hail.log'))

if not os.path.isfile(args.vcf):
    logger.error("Could not find vcf file at {}".format(args.vcf))
    sys.exit(1)

def get_dnm_loci(dnm_file1, dnm_file2, index_id):
    # read DNM files
    dnm1 = pd.read_csv(dnm_file1, sep="\t")
    dnm2 = pd.read_csv(dnm_file2, sep="\t")

    # if locus exported as one field of form contig:position, split into two fields
    if "locus" in dnm1:
        dnm1[['locus.contig', 'locus.position']] = dnm1['locus'].str.split(':', 1, expand=True)
    if "locus" in dnm2:
        dnm2[['locus.contig', 'locus.position']] = dnm2['locus'].str.split(':', 1, expand=True)
    
    dnm1 = dnm1.dropna(subset=["locus.position", "locus.contig"])
    dnm2 = dnm2.dropna(subset=["locus.position", "locus.contig"])
    dnm1['locus.position'] = dnm1['locus.position'].astype(int)
    dnm2['locus.position'] = dnm2['locus.position'].astype(int)

    # filter for the ID of the genome
    dnm1 = dnm1[dnm1.s==index_id][['locus.contig', 'locus.position']]
    dnm2 = dnm2[dnm2.s==index_id][['locus.contig', 'locus.position']]

    # overlap between the two
    dnm_overlap = pd.merge(dnm1, 
                         dnm2, 
                         how='inner', 
                         on=['locus.contig', 'locus.position'])

    dnm_difference = pd.concat([dnm1,dnm2]).drop_duplicates(keep=False)
    
    # create a list of the loci of putative DNMs
    in_overlap = [hl.locus(contig, position) 
                for contig, position in zip(dnm_overlap["locus.contig"], dnm_overlap["locus.position"])]
    
    outside_overlap = [hl.locus(contig, position) 
                for contig, position in zip(dnm_difference["locus.contig"], dnm_difference["locus.position"])]
    
    return in_overlap, outside_overlap

def annotate_labels_from_gt(data_mt, index_id):
    data_mt = data_mt.annotate_rows(
        info=data_mt.info.annotate(
            label = hl.str(
                hl.agg.filter(
                    data_mt.s == index_id,
                    hl.agg.collect(data_mt.GT.n_alt_alleles())[0]))))
    return data_mt


def filter_genotypes(data_mt, index_id, contigs):
    
    # keep only the column of the index, so there are no conflicts in make_examples
    data_mt = data_mt.filter_cols(data_mt.s==index_id)

    # filter out those loci for which GT is missing in the index
    data_mt = data_mt.filter_rows(hl.is_nan(hl.float(data_mt.info.label)).__invert__())
    
    # keep only specified contigs
    data_mt = data_mt.filter_rows(hl.set(contigs).contains(data_mt.locus.contig))
    
    return data_mt

def random_subset(data_mt, subset_size):
    sampling_prob = min(subset_size / data_mt.count()[0], 1.0)
    return data_mt.sample_rows(sampling_prob)

def main():    
    full_mt = hl.import_vcf(args.vcf,
                            array_elements_required=False,
                            find_replace=("-inf", "-100"),
                            skip_invalid_loci=True,)
    
    # filter SNPs
    full_mt = full_mt.filter_rows((full_mt.alleles[0].length()==1) & (full_mt.alleles[1].length()==1))
    non_dnm_mt = full_mt

    # MSDN
    try:
        in_overlap_msdn, _ = get_dnm_loci(args.msdn1, 
                                args.msdn2, 
                                args.index_id)
        
        # exlude the putative MSDNs from the full_mt, so that the model is never trained on MSDNs with label 1 instead of 4
        non_dnm_mt = non_dnm_mt.filter_rows(hl.set(in_overlap_msdn).contains(non_dnm_mt.locus).__invert__())

        # subset the VCF
        msdn_mt = full_mt.filter_rows(hl.set(in_overlap_msdn).contains(full_mt.locus))
        # msdn gets label "4"
        msdn_mt = msdn_mt.annotate_rows(info=msdn_mt.info.annotate(label="4"))
        in_overlap_msdn = hl.set(in_overlap_msdn)

        print("got {} overlapping msdns".format(in_overlap_msdn.length()))

    except Exception as err:
        print("got exception in msdn processing: ", err)
        in_overlap_msdn = hl.expr.functions.empty_set(hl.expr.types.tlocus())

    # DNMs
    try:
        in_overlap_dnm, outside_overlap = get_dnm_loci(args.dnm1, 
                                args.dnm2, 
                                args.index_id)

        print("got {} overlapping dnms".format(len(in_overlap_dnm)))

        in_overlap_dnm_not_msdn = hl.set(in_overlap_dnm).difference(in_overlap_msdn)

        # exlude the putative DNMs from the full_mt, so that the model is never trained on DNMs with label 1 instead of 3
        non_dnm_mt = non_dnm_mt.filter_rows(in_overlap_dnm_not_msdn.contains(non_dnm_mt.locus).__invert__())

        # subset the VCF
        dnm_mt = full_mt.filter_rows(in_overlap_dnm_not_msdn.contains(full_mt.locus))
        # dnm gets label "3"
        dnm_mt = dnm_mt.annotate_rows(info=dnm_mt.info.annotate(label="3"))

        print("got {} dnms not in msdn callset".format(in_overlap_dnm_not_msdn.length()))
    except Exception as err:
        print("got exception in dnm processing: ", err) 
        in_overlap_dnm_not_msdn = hl.expr.functions.empty_set(hl.expr.types.tlocus())
        
    # annotate the matrix table with class labels "0", "1", "2" (depending on index genotype)
    non_dnm_mt = annotate_labels_from_gt(non_dnm_mt, args.index_id)
    
    # get subsets with other labels, roughly matching args.prop_std_classes times number of DNMs each
    if hl.eval(in_overlap_dnm_not_msdn.length()) != 0:
        n_dnm = dnm_mt.rows().count()
        label0_mt = random_subset(non_dnm_mt.filter_rows(non_dnm_mt.info.label=="0"), args.prop_std_classes * n_dnm)
        label1_mt = random_subset(non_dnm_mt.filter_rows(non_dnm_mt.info.label=="1"), args.prop_std_classes * n_dnm)
        label2_mt = random_subset(non_dnm_mt.filter_rows(non_dnm_mt.info.label=="2"), args.prop_std_classes * n_dnm)
        
        # join DNMs to the other classes (automatically sorted by contig and position)
        all_classes_mt = dnm_mt.union_rows(label0_mt).union_rows(label1_mt).union_rows(label2_mt)
        
        # use DNM candidates outsite overlap of the two callers as label 0, 1 or 2 examples if INDEL is within a defined window (those are very likely calling artifacts and thus candidates for false positives)
        if args.indel_window_file:
            fp_around_indels_mt = hl.import_vcf(args.indel_window_file,
                                                array_elements_required=False,
                                                find_replace=("-inf", "-100"),
                                                force_bgz=True,
                                                skip_invalid_loci=True,)
            # filter SNPs
            fp_around_indels_mt = fp_around_indels_mt.filter_rows((fp_around_indels_mt.alleles[0].length()==1) & (fp_around_indels_mt.alleles[1].length()==1))

            fp_around_indels_mt = fp_around_indels_mt.filter_rows(hl.set(outside_overlap).contains(fp_around_indels_mt.locus))
            fp_around_indels_mt = annotate_labels_from_gt(fp_around_indels_mt, args.index_id)
            all_classes_mt = all_classes_mt.union_rows(fp_around_indels_mt)
        else:
            logger.warning("No VCF file with variant sites in windows around indels provided; DNM candidates outside overlap of the two callers will be ignored.")
    else:
        logger.warning("got no dnms that are not part of msdn clusters")
        all_classes_mt = non_dnm_mt

    # join MSDNs
    if hl.eval(in_overlap_msdn.length()) != 0:
        # append MSDNs to output data
        all_classes_mt = all_classes_mt.union_rows(msdn_mt)
        # label all DNMs around MSDN with label 4
        msdn_interval_list = []
        msdns = in_overlap_msdn.collect()
        for msdn in msdns:
            contig = list(msdn)[0].contig
            position = list(msdn)[0].position
            msdn_interval_list.append({"interval_str": str(contig) + ":" + str(position-20) + "-" + str(position+20)})

        msdn_interval_table = hl.Table.parallelize(
            hl.literal(msdn_interval_list, 'array<struct{interval_str:str}>'))
        msdn_interval_table = msdn_interval_table.transmute(interval=hl.parse_locus_interval(msdn_interval_table.interval_str)).key_by('interval')

        all_classes_mt = all_classes_mt.annotate_rows(
            info=all_classes_mt.info.annotate(
                label = hl.case()
                .when((hl.is_defined(msdn_interval_table[all_classes_mt.locus])) & (all_classes_mt.info.label=="3"), "4")
                .default(all_classes_mt.info.label)))
        
    # filter the matrix table
    all_classes_mt = filter_genotypes(all_classes_mt, args.index_id, args.contigs)
    
    # join file with sites at which caller identified DNMs in first round, but that are likely artifacts
    if args.first_round_file:
        fp_first_round_mt = hl.import_vcf(args.first_round_file,
                                            array_elements_required=False,
                                            find_replace=("-inf", "-100"),
                                            force_bgz=True,
                                            skip_invalid_loci=True,)


        all_classes_mt = all_classes_mt.union_rows(fp_first_round_mt)
        
    # export truth VCF
    hl.export_vcf(all_classes_mt, args.out_dir + "/" + args.index_id + ".vcf")

if __name__ == "__main__":
    main()
