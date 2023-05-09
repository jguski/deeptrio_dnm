from absl import logging 
import argparse
import os
import sys
import json
import config
import glob
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to silence excessive TF output
import tensorflow as tf
from utils.model_utils import create_inception_v3, init_callbacks
from utils.io_utils import get_dataset, get_train_val_datasets, get_cv_datasets


log_level = {
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

parser = argparse.ArgumentParser(
    prog="RETRAIN DV/DT MODEL",
    description='Retrain a model provided by DeepVariant/DeepTrio.')

parser.add_argument('--train_data_path', type=str, required=True, help='The path of the pile-up images for training/validation.')
parser.add_argument('--val_data_path', type=str, default=None, help='The path of the pile-up images for validation. If specified, no train/val split is created by this script and parameter --val_split is ignored.')

parser.add_argument('--input_shape', type=tuple, default=config.network_shape['input_shape'], help='The expected shape of input tensors (h,w,c).')
parser.add_argument('--n_classes', type=int, default=config.network_shape['n_classes'], help='The number of neurons in output layer.')
parser.add_argument('--class_weights', nargs='+', default=None, help='A list of weights for all classes (length must match --n_classes).')
parser.add_argument('--batch_size', type=int, default=config.train_params['batch_size'], help='The number of samples in each batch.')
parser.add_argument('--epochs', type=int, default=config.train_params['epochs'], help='The number of epochs for training.')
parser.add_argument('--val_split', type=float, default=config.train_params['val_split'], help='Fraction of the training data to be used as validation data.')
parser.add_argument('--n_folds', type=int, default=config.train_params['n_folds'], help='Number of folds for cross-validation. If specified, parameters --val_split and --val_data_path are ignored.')
parser.add_argument('--learning_rate', type=float, default=config.train_params['learning_rate'], help='Initial learning rate.')
parser.add_argument('--rmsprop_momentum', type=float, default=config.train_params['rmsprop_momentum'], help='Momentum.')
parser.add_argument('--rmsprop_decay', type=float, default=config.train_params['rmsprop_decay'], help='Decay term for RMSProp.')
parser.add_argument('--rmsprop_epsilon', type=float, default=config.train_params['rmsprop_epsilon'], help='Epsilon term for RMSProp.')
parser.add_argument('--learning_rate_decay_factor', type=float, default=config.train_params['learning_rate_decay_factor'], help='Learning rate decay factor.')
parser.add_argument('--num_epochs_per_decay', type=float, default=config.train_params['num_epochs_per_decay'], help='Number of epochs after which learning rate decays.')
parser.add_argument('--log_level', type=str, default=config.general_config['log_level'], help='Verbosity level ("error", "warning", "info", "debug").')
parser.add_argument('--model_path', type=str, default=config.general_config['model_path'], help='The path of the model checkpoint to be retrained.')
parser.add_argument('--variable_path', type=str, default=config.general_config['variable_path'], help='The path of a .txt-file containing model variable names to be loaded into the model (needed if checkpoint is name-based as the ones that come with DeepVariant/DeepTrio).')
parser.add_argument('--log_dir', type=str, default=config.general_config['log_dir'], help='Path to write Tensorboard-related logs during training.')
parser.add_argument('--checkpoint_dir', type=str, default=config.general_config['checkpoint_dir'], help='Path to write model checkpoints during training.')
parser.add_argument('--save_checkpoint_every', type=int, default=config.general_config['save_checkpoint_every'], help='Number of batches after which checkpoint is saved.')
parser.add_argument('--shuffle_buffer', type=int, default=config.general_config['shuffle_buffer'], help='Window of examples to draw next from for shuffling.')
parser.add_argument('--freeze_hidden_layers', action='store_true', help='If true: Freezing all but the output layer.')
parser.add_argument('--combine_labels', nargs='+', type=str, default=None, help='A list of tuples a,b. If label a is present in an example, then label b will also be active. Example: --combine labels 5,4 4,5')
parser.add_argument('--no_early_stopping', dest='early_stopping', default=True, action='store_false', help='Apply early stopping criteria for training.')


args = parser.parse_args()

# write all parameters to a file
with open(args.log_dir + '/params.json', 'w') as f:
    json.dump(vars(args), f)
    
# initialize dictionary for class weights
if not args.class_weights:
    args.class_weights = args.n_classes * [1.0]
elif len(args.class_weights) != args.n_classes:
    logging.error("Number of classes ({}) does not match length of vector with class weights ({}).".format(args.n_classes, len(args.class_weights)))
    sys.exit(1)
class_weights_dict = dict(zip(range(args.n_classes), list(map(float, args.class_weights))))
# convert input of args.combine_labels to a list of tuples
if args.combine_labels:
    combine_labels = [tuple(map(int, cl.split(','))) for cl in args.combine_labels]
else:
    combine_labels = None

if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

def main():
    
    if not args.n_folds:
        if args.val_data_path:
            
            ###### CASE 1 ##############################################################################
            ###### Training and validation data explicitly provided in two directories #################
            ############################################################################################
            
            # list all subdirectories of train_data_path and val_data_path (this should be the directories with examples for the individual samples)
            train_samples = np.asarray(glob.glob(args.train_data_path + "/*")) 
            val_samples = np.asarray(glob.glob(args.val_data_path + "/*")) 
            # make sure that only directories are used
            train_samples = train_samples[np.asarray(list(map(os.path.isdir, train_samples)))]
            val_samples = val_samples[np.asarray(list(map(os.path.isdir, val_samples)))]
            
            dataset_train, nexamples_train = get_dataset(train_samples, 
                                                        args.batch_size,
                                                        args.input_shape,
                                                        args.n_classes,
                                                        args.shuffle_buffer,
                                                        purpose='training',
                                                        training=True,
                                                        combine_labels=combine_labels) 
            
            dataset_val, nexamples_val = get_dataset(val_samples, 
                                                        args.batch_size,
                                                        args.input_shape,
                                                        args.n_classes,
                                                        args.shuffle_buffer,
                                                        purpose='validation',
                                                        training=True,
                                                        combine_labels=combine_labels) 
        else:
            ###### CASE 2 ###################################################################################
            ###### All data in one directory and no split provided: Create split at runtime #################
            #################################################################################################
            
            dataset_train, nexamples_train, dataset_val, nexamples_val = get_train_val_datasets(args.train_data_path, 
                                                                                                   args.batch_size,
                                                                                                   args.input_shape,
                                                                                                   args.n_classes,
                                                                                                   args.shuffle_buffer,
                                                                                                   val_split=args.val_split,
                                                                                                   combine_labels=combine_labels) 
        # create inception V3 model with given parameters    
        model = create_inception_v3(args.input_shape,
                                    args.n_classes,
                                    args.model_path,
                                    args.variable_path,
                                    False,
                                    args.freeze_hidden_layers,
                                    args.learning_rate,
                                    args.num_epochs_per_decay,
                                    args.learning_rate_decay_factor,
                                    args.rmsprop_momentum,
                                    args.rmsprop_epsilon,
                                    args.rmsprop_decay)
        
        # define callbacks
        callbacks = init_callbacks(checkpoint_dir=args.checkpoint_dir,
                                       checkpoint_filename='best',
                                       logs_dir=args.log_dir,
                                       early_stopping=args.early_stopping)

        # training loop
        logging.info("Starting training loop.")
        model.fit(dataset_train, 
              validation_data=dataset_val,  
              epochs=args.epochs,
              steps_per_epoch=nexamples_train//args.batch_size,
              validation_steps=nexamples_val//args.batch_size,
              callbacks=callbacks,
              verbose=2,
              class_weight=class_weights_dict)

        # save final model with structure (not just weights)
        model.save(args.checkpoint_dir)
            
    else:
        
        ###### CASE 3 ####################################################################################
        ###### All data in one directory and no split provided: k-fold cross validation ##################
        ##################################################################################################
        
        cv_data_generator = get_cv_datasets(args.train_data_path, 
                                            args.batch_size,
                                            args.input_shape,
                                            args.n_classes,
                                            args.shuffle_buffer,
                                            n_folds=args.n_folds,
                                            combine_labels=combine_labels)
        
        fold = 0
        for dataset_train, nexamples_train, dataset_val, nexamples_val in cv_data_generator:
            # create inception V3 model with given parameters    
            model = create_inception_v3(args.input_shape,
                                        args.n_classes,
                                        args.model_path,
                                        args.variable_path,
                                        False,
                                        args.freeze_hidden_layers,
                                        args.learning_rate,
                                        args.num_epochs_per_decay,
                                        args.learning_rate_decay_factor,
                                        args.rmsprop_momentum,
                                        args.rmsprop_epsilon,
                                        args.rmsprop_decay)
            # define callbacks     
            callbacks = init_callbacks(checkpoint_dir=args.checkpoint_dir,
                                       checkpoint_filename='best_0'+str(fold),
                                       logs_dir=args.log_dir)
        
            # training loop
            logging.info("Starting training loop for fold {}.".format(fold))
            model.fit(dataset_train, 
                  validation_data=dataset_val,  
                  epochs=args.epochs,
                  steps_per_epoch=nexamples_train//args.batch_size,
                  validation_steps=nexamples_val//args.batch_size,
                  callbacks=callbacks,
                  verbose=2,
                  class_weight=class_weights_dict)
            model.save(args.checkpoint_dir + "/model_fold_" + str(fold))
            fold+=1
    
    
if __name__ == "__main__":
    logging.set_verbosity(log_level[args.log_level])
    
    main()
