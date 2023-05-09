import tensorflow as tf
import glob
import argparse

def create_pbtxt(sample_path,
                tfrecord_pattern="examples.tfrecord-?????-of-?????.gz",
                dataset_config_pbtxt="dataset_config.pbtxt"):
    
    pbtxt_path = sample_path + "/" + dataset_config_pbtxt
    tfrecords = glob.glob(sample_path + "/" + tfrecord_pattern)
    num_examples = sum(1 for _ in tf.data.TFRecordDataset(tfrecords, compression_type="GZIP", num_parallel_reads=-1))
    with open(pbtxt_path, 'w') as f:
        f.write("tfrecord_path: " + str(sample_path) + "/" + tfrecord_pattern + "\n")
        f.write("num_examples: " + str(num_examples) + "\n")
    return [pbtxt_path]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="CREATE DATASET CONFIG PBTXT",
        description='Create a dataset config .pbtxt-file (containing tfrecord file pattern and number of examples)')

    parser.add_argument('--sample_path', type=str, required=True, help='Path of a directory containing tfrecords with examples.')
    parser.add_argument('--tfrecord_pattern', type=str, default="examples.tfrecord-?????-of-?????.gz", help='Pattern of the tfrecord files within this directory.')
    parser.add_argument('--dataset_config_pbtxt', type=str, default="dataset_config.pbtxt", help='Name of the config .pbtxt-file to write to --sample_path.')

    args = parser.parse_args()

    create_pbtxt(args.sample_path,
                args.tfrecord_pattern,
                args.dataset_config_pbtxt)