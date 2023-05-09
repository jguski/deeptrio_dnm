import numpy as np
import math
import pysam
import argparse
import hail as hl

parser = argparse.ArgumentParser(
    prog="DATA AUGMENTATION",
    description='Insert synthetic MSDNs (or clustered inherited variants) in regions with DNMs.')

parser.add_argument('--original_bam_child', type=str, help="BAM file of the original alignment for child.")
parser.add_argument('--original_bam_parent1', type=str, help="BAM file of the original alignment for parent 1.")
parser.add_argument('--original_bam_parent2', type=str, help="BAM file of the original alignment for parent 2.")
parser.add_argument('--original_vcf', type=str, help="Truth VCF with annotated labels (label=3 for DNM).")
parser.add_argument('--output_directory', type=str, help="Directory where new (unsorted) BAMs and VCF will be written.")

args = parser.parse_args()

hl.init()

def exponential_probabilities(distances, scale=1):
    e_transform = (1/scale)*math.e**(-(1/scale)*distances)
    return e_transform / sum(e_transform)

def find_new_bases(original_alignment, contig, pileup_positions):
    bases_for_shifts = {k: [] for k in pileup_positions}

    # decide which bases the original bases will be corrupted with
    for pileupcolumn in original_alignment.pileup(contig, pileup_positions[0], pileup_positions[-1]):
        for shift in pileup_positions:
            if pileupcolumn.pos == shift:
                for pileupread in pileupcolumn.pileups:
                        bases_for_shifts[shift] += pileupread.alignment.query_sequence[pileupread.query_position]
    original_bases = {k: max(set(bases_for_shifts[k]), key = bases_for_shifts[k].count) for k in bases_for_shifts.keys()}
    corrupt_with = np.array([np.random.choice(list(set(["A","C","G","T"])-set(original_bases[k]))) for k in original_bases.keys()])
    
    return corrupt_with, original_bases


def manipulate_reads(original_alignment, contig, pileup_positions, new_bases, p_manipulate, inherited=True):
    manipulated_reads = {}
    
    for pileupcolumn in original_alignment.pileup(contig, pileup_positions[0], pileup_positions[-1]):
        for shift in pileup_positions:
            if pileupcolumn.pos == shift:
                for pileupread in pileupcolumn.pileups:
                    if pileupread.alignment.query_name not in manipulated_reads.keys():
                        # decide if the read will support the artificial MSDN
                        supports_msdn = np.random.choice([True,False], p=[p_manipulate, 1-p_manipulate])
                        if supports_msdn:
                                # determine length of the read
                                read_length = len(pileupread.alignment.query_sequence)

                                # filter only those MSDN positions that are in the read (and not within 5 bp of the read's ends)
                                manipulate_positions_all = np.array((pileup_positions - pileupread.alignment.reference_start))
                                positions_to_keep = np.where((manipulate_positions_all>4) & (manipulate_positions_all<(read_length-5)) & inherited)
                                manipulate_positions = manipulate_positions_all[positions_to_keep]
                                corrupt_positions_with = new_bases[positions_to_keep]

                                # manipulate sequence and qualities
                                manipulated_read = pileupread.alignment
                                manipulated_sequence = np.array(list(manipulated_read.query_sequence))
                                manipulated_qualities = np.array(manipulated_read.query_qualities)
                                manipulated_sequence[manipulate_positions] = corrupt_positions_with
                                manipulated_qualities[manipulate_positions] = 37  

                                # save the manipulated read (to be written to new bam file later)
                                manipulated_read.query_sequence = "".join(manipulated_sequence)
                                manipulated_read.query_qualities = manipulated_qualities
                                manipulated_reads[pileupread.alignment.query_name] = manipulated_read

                        else:
                            # keep original read
                            manipulated_reads[pileupread.alignment.query_name] = pileupread.alignment
    return manipulated_reads
                            
def main():
    
    # extract VCF header for the output VCF
    vcf_header = pysam.VariantFile(args.original_vcf).header
    
    # get all variants labelled as DNM from the original VCF
    original_variants = hl.import_vcf(args.original_vcf, array_elements_required=False, find_replace=("-inf", "-100"), force_bgz=True)
    dnms = original_variants.filter_rows(original_variants.info.label=="3").locus.take(100000)
    
    # import the original alignment files
    original_alignment_child = pysam.AlignmentFile(args.original_bam_child, "rb")
    original_alignment_parent1 = pysam.AlignmentFile(args.original_bam_parent1, "rb")
    original_alignment_parent2 = pysam.AlignmentFile(args.original_bam_parent2, "rb")
    
    # the output BAM
    output_bam_child = pysam.AlignmentFile(args.output_directory + "/synthetic_MSDN_child_unsorted.bam", "w", header=original_alignment_child.header)
    output_bam_parent1 = pysam.AlignmentFile(args.output_directory + "/synthetic_MSDN_parent1_unsorted.bam", "w", header=original_alignment_parent1.header)
    output_bam_parent2 = pysam.AlignmentFile(args.output_directory + "/synthetic_MSDN_parent2_unsorted.bam", "w", header=original_alignment_parent2.header)
    output_vcf = pysam.VariantFile(args.output_directory + "/synthetic_MSDN_unsorted.vcf", "w", header=vcf_header)
    
    for dnm in dnms:
        # randomly draw the number of synthetic MSDN, their offset from the original DNM, if the offset is to its left or right, and the shifts between the MSDN in the cluster
        msdn_size = np.random.choice(range(2,5), p=exponential_probabilities(np.arange(3)))
        msdn_offset = np.random.choice(list(range(150,200)))
        left_or_right = np.random.choice([-1, 1])
        msdn_shifts = [0] + np.random.choice(np.arange(1,20), msdn_size-1, p=exponential_probabilities(np.arange(1,20), scale=5), replace=False).tolist()
        
        try:
            # determine the positions of the pileups and the new bases at these positions
            pileup_positions = dnm.position + (left_or_right*msdn_offset) + msdn_shifts
            new_bases, original_bases = find_new_bases(original_alignment_child, dnm.contig, pileup_positions)
        except:
            continue
        
        # decide which variants in the cluster are inherited and from which parent
        inherited = np.random.choice([False,True], msdn_size, p=[0.7,0.3])
        inherited_from_parent1 = np.random.choice([False,True])
        heterozyguous_in_parents = np.random.choice([False,True], p=[0.9,0.1])
        
        # create manipulated reads for child and parents
        if heterozyguous_in_parents:
            # in this case, both parents are HET but the child is HOM
            reads_to_write_child = manipulate_reads(original_alignment_child, dnm.contig, pileup_positions, new_bases, 0)
            reads_to_write_parent1 = manipulate_reads(original_alignment_parent1, dnm.contig, pileup_positions, new_bases, p_manipulate=0.5)
            reads_to_write_parent2 = manipulate_reads(original_alignment_parent2, dnm.contig, pileup_positions, new_bases, p_manipulate=0.5)
        elif inherited_from_parent1:
            # in this case, the variants selected in inherited are also present in parent1
            reads_to_write_child = manipulate_reads(original_alignment_child, dnm.contig, pileup_positions, new_bases, 0.5)
            reads_to_write_parent1 = manipulate_reads(original_alignment_parent1, dnm.contig, pileup_positions, new_bases, p_manipulate=0.5, inherited=inherited)
            reads_to_write_parent2 = manipulate_reads(original_alignment_parent2, dnm.contig, pileup_positions, new_bases, p_manipulate=0)
        else:
            # in this case, the variants selected in inherited are also present in parent2
            reads_to_write_child = manipulate_reads(original_alignment_child, dnm.contig, pileup_positions, new_bases, 0.5)
            reads_to_write_parent1 = manipulate_reads(original_alignment_parent1, dnm.contig, pileup_positions, new_bases, p_manipulate=0)
            reads_to_write_parent2 = manipulate_reads(original_alignment_parent2, dnm.contig, pileup_positions, new_bases, p_manipulate=0.5, inherited=inherited)
            
        # write to new BAM
        for read in reads_to_write_child.values():
            output_bam_child.write(read)
        for read in reads_to_write_parent1.values():
            output_bam_parent1.write(read)
        for read in reads_to_write_parent2.values():
            output_bam_parent2.write(read)
        
        # write to VCF (with labels)
        # base label: 1 (HET)
        labels = np.full(msdn_size, '1')
        if heterozyguous_in_parents:
            # no variant in child -> label as 0 (HOM)
            labels = np.full(msdn_size, '0')
        elif sum(np.invert(inherited))>1:
            # if more than one DNM in cluster: Label those DNMs as 4 (MSDN)
            labels[np.invert(inherited)] = '4'
        else:
            # if just one DNM in cluster: Treat it as a standard DNM (label 3)
            labels[np.invert(inherited)] = '3'
        
        for i in range(msdn_size):
 
            new_record = vcf_header.new_record(contig=dnm.contig,
                start=pileup_positions[i],
                stop=pileup_positions[i]+1,
                alleles=(original_bases[pileup_positions[i]], new_bases[i]),
                info={"label": labels[i]})
            
            if heterozyguous_in_parents:
                new_record.samples[vcf_header.samples[0]]['GT'] = (0,0)
            else:
                # all other synthetic variants are HET
                new_record.samples[vcf_header.samples[0]]['GT'] = (0,1)            
            
            output_vcf.write(new_record)
        

if __name__ == "__main__":
    main()
