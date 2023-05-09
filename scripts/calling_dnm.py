from absl import logging 
import argparse
import config
import os
import glob
import numpy as np
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to silence excessive TF output
import tensorflow as tf
from utils.model_utils import load_weights_layerwise
from utils.io_utils import get_dataset, to_table_row
from utils.protos import deepvariant_pb2
from utils.protos.third_party.nucleus.protos import variants_pb2

log_level = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

parser = argparse.ArgumentParser(
    prog="CALL DE-NOVO MUTATIONS",
    description='Use a trained model to call de-novo mutations in given examples.')

# required
parser.add_argument('--call_data_path', type=str, required=True, help='The path of the pile-up images to call de-novo mutations.')
parser.add_argument('--model_path', type=str, required=True, help='The path of the model checkpoint to be be used for calling.')
parser.add_argument('--call_file', type=str, required=True, help='The path where tsv with de-novo calls will be written.')
parser.add_argument('--dnm_threshold', type=float, required=True, help='Threshold of probability to use to decide whether a variant is actually de novo.')

# fine to run with default
parser.add_argument('--msdn_threshold', type=float, default=0.4, help='Threshold of probability to use to decide whether a variant is actually an MSDN.')
parser.add_argument('--input_shape', type=tuple, default=config.network_shape['input_shape'], help='The expected shape of input tensors (h,w,c).')
parser.add_argument('--n_classes', type=int, default=4, help='The number of neurons in output layer.')
parser.add_argument('--batch_size', type=int, default=config.train_params['batch_size'], help='The number of samples in each batch.')
parser.add_argument('--dnm_class', type=int, default=3, help='Expected de novo class.')
parser.add_argument('--msdn_class', type=int, default=4, help='Expected MSDN class.')
parser.add_argument('--log_level', type=str, default=config.general_config['log_level'], help='Verbosity level ("error", "warning", "info", "debug").')
parser.add_argument('--variable_path', type=str, default=config.general_config['variable_path'], help='The path of a .txt-file containing model variable names to be loaded into the model (needed if checkpoint is name-based as the ones that come with DeepVariant/DeepTrio).')

args = parser.parse_args()
 

def main():
    
   # list all subdirectories of eval_data_path (this should be the directories with examples for the individual samples)
    calling_samples = np.asarray(glob.glob(args.call_data_path + "/*")) 
    # make sure that only directories are used
    calling_samples = calling_samples[np.asarray(list(map(os.path.isdir, calling_samples)))]
    
    # create dataset from given config .pbtxt-file
    dataset_calling, nexamples_calling = get_dataset(calling_samples, 
                                                        args.batch_size,
                                                        args.input_shape,
                                                        args.n_classes,
                                                        purpose='calling',
                                                        training=False,
                                                        calling=True) 

    # load basic inception_v3 model
    model=tf.keras.applications.InceptionV3(
        include_top=True, 
        weights=None, 
        input_shape=args.input_shape, 
        classes=args.n_classes
    )
     
    try:
        # load model from file
        model = tf.keras.models.load_model(args.model_path)
        
    except:
        logging.warning("Loading model weights with tf.keras.models.load_model(..) failed. Restoring weights layerwise instead.")

        # load basic inception_v3 model
        model=tf.keras.applications.InceptionV3(
            include_top=True, 
            weights=None, 
            input_shape=args.input_shape, 
            classes=args.n_classes
        )
        load_weights_layerwise(model, 
                           args.model_path,
                           args.variable_path,
                           output_layer=True,
                           eval_mode=True)


    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
    
                    
    # create a list for de_novos
    dnm_loci = []
    
    logging.info("Starting calling loop.")
    
    file = open(args.call_file, 'w', newline ='') 
    with file:  
        write = csv.writer(file) 
        #write.writerow(("Locus", "Probability_DNM", "Ref_allele", "Alt_alleles", "VAF", "DP"))
        write.writerow(("Locus", "Probability_DNM", "Probability MSDN", "Ref_allele", "Alt_alleles", "VAF", "DP"))    
    
    cnt=0
    for examples in dataset_calling:         
        # forward pass through model
        net_output = model.predict(examples[0])
        
        output = to_table_row(
            examples,
            net_output,
            dnm_threshold=args.dnm_threshold,
            msdn_threshold=args.msdn_threshold,
            dnm_class=args.dnm_class,
            msdn_class=args.msdn_class,
        )
        
        # update call_file with DNMs predicted in the currect batch 
        file = open(args.call_file, 'a', newline ='') 
        with file:     
            write = csv.writer(file) 
            write.writerows(output)

           
        
if __name__ == "__main__":
    logging.set_verbosity(log_level[args.log_level])
    
    main()
