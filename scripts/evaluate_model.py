import argparse
import config
import os
import glob
import numpy as np
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to silence excessive TF output
import tensorflow as tf
from sklearn.metrics import classification_report
from utils.model_utils import load_weights_layerwise
from utils.io_utils import get_dataset
import logging

logger = logging.getLogger('evaluate_model')
log_level = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

parser = argparse.ArgumentParser(
    prog="EVALUATE (RETRAINED) DV/DT MODEL",
    description='Evaluate a model provided by DeepVariant/DeepTrio or a retrained one.')

# required
parser.add_argument('--eval_data_path', type=str, required=True, help='The path of the pile-up images for evaluation.')
parser.add_argument('--model_path', type=str, required=True, help='The path of the model checkpoint to be evaluated.')
parser.add_argument('--output_path', type=str, required=True, help='The path to write the evaluation metrics.')

# fine to run with default
parser.add_argument('--input_shape', type=tuple, default=config.network_shape['input_shape'], help='The expected shape of input tensors (h,w,c).')
parser.add_argument('--n_classes', type=int, default=config.network_shape['n_classes'], help='The number of neurons in output layer.')
parser.add_argument('--batch_size', type=int, default=config.train_params['batch_size'], help='The number of samples in each batch.')
parser.add_argument('--log_level', type=str, default=config.general_config['log_level'], help='Verbosity level ("error", "warning", "info", "debug").')
parser.add_argument('--variable_path', type=str, default=config.general_config['variable_path'], help='The path of a .txt-file containing model variable names to be loaded into the model (needed if checkpoint is name-based as the ones that come with DeepVariant/DeepTrio).')

args = parser.parse_args()
 

def main():
    
   # list all subdirectories of eval_data_path (this should be the directories with examples for the individual samples)
    eval_samples = np.asarray(glob.glob(args.eval_data_path + "/*")) 
    # make sure that only directories are used
    eval_samples = eval_samples[np.asarray(list(map(os.path.isdir, eval_samples)))]

    # create dataset from given config .pbtxt-file
    dataset_eval, nexamples_eval = get_dataset(eval_samples,
                                                        args.batch_size,
                                                        args.input_shape,
                                                        args.n_classes,
                                                        purpose='evaluation',
                                                        training=False) 

    
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
        logger.warning("Loading model weights with tf.keras.models.load_model(..) failed. Restoring weights layerwise instead.")

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
    
    
    logger.info("Starting evaluation loop.")
    #results = model.evaluate(dataset_eval, 
    #      steps=nexamples_eval//args.batch_size,       
    #      verbose=2)
    
    #print("test loss, test acc:", results)
    
    y_true, y_predicted = [], []
    pbar = tqdm(total=nexamples_eval)
    for d in dataset_eval:
        # read true labels from examples
        y_true += list(np.argmax(d[1].numpy(), axis=1))
        # predict classes with model
        y_predicted += list(np.argmax(model.predict(d[0]), axis=1))
        pbar.update(args.batch_size)

    pbar.close()


    results = classification_report(y_true, y_predicted)

    out_path = None
    if os.path.isdir(args.output_path):
        out_path = args.output_path + '/evaluation_' + args.eval_data_path.split("/")[-1] + '.txt'
    else:
        out_path = args.output_path
    
    with open(out_path, 'w') as fp:
        fp.write(results)

    
if __name__ == "__main__":
    logger.setLevel(log_level[args.log_level])
    
    main()
