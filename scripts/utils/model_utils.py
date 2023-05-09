import tensorflow as tf
from absl import logging 

def get_weight_matrices(checkpoint_path):
    """ Extracts all weight matrices from a (TF1) checkpoint from given path
        Args:
          checkpoint_path: Path to the (TF1) checkpoint.

        Returns:
          Dictionary with pairs of variable  names as keys and the corresponding weight matrices from checkpoint.
    """
    weight_matrices = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
          weight_matrices[key] = tf.Variable(reader.get_tensor(key))
    return weight_matrices
    

def load_weights_layerwise(model, 
                           checkpoint_path, 
                           variables_file=None, 
                           input_layer=True, 
                           output_layer=False, 
                           eval_mode=False):
    """ Loads weight matrices into a model layer by layer, accessing variables in a checkpoint by name.

        Args:
          model: A tensorflow.python.keras.engine.training.Model instance
          checkpoint: A (TF1) checkpoint path
          variables_file: Link to txt-file with ordered list of variable names (weight matrices to extract)
          input_layer: Determines whether weights for first layer are being loaded
          output_layer: Determines whether weights for final layer are being loaded
          eval_mode: Specifies whether exponential moving average weights are loaded from checkpoint instead of normal weight matrices
          

        Returns:
          Model with loaded weight matrices.
    """
    # load unordered weight matrices from checkpoint
    weights=get_weight_matrices(checkpoint_path)
    
    # load ordered list of variables from txt-file
    with open(variables_file) as file:
        variable_names=file.read().splitlines()
    
    # iterate only over trainable layers (nothing to be loaded for the others)
    trainable_layers=[layer for layer in model.layers if len(layer.weights)>0]
    
    if eval_mode:
        var_extension="/ExponentialMovingAverage"
    else:
        var_extension=""
    
    compat_error=False

    for i, layer in enumerate(trainable_layers):
        
        # skip input and output layer if specified in args
        if i==0 and not input_layer:
            logging.info("Weights not loaded for input layer.")
            continue
        if i==(len(trainable_layers)-1) and not output_layer:
            logging.info("Weights not loaded for output layer.")
            break
            
        try:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.set_weights([weights[variable_names[i] + "/weights" + var_extension].numpy()[0][0], 
                                  weights[variable_names[i] + "/biases" + var_extension].numpy()])
            elif "batch_normalization" in layer.name:
                if eval_mode:
                    layer.set_weights([weights[variable_names[i] + "/beta" + var_extension].numpy(), 
                                      weights[variable_names[i] + "/moving_mean"].numpy(),
                                      weights[variable_names[i] + "/moving_variance"].numpy()])
            else:
                layer.set_weights([weights[variable_names[i]+"/weights" + var_extension].numpy()])
                
        except ValueError as e:
            logging.error("Compatibility problem between {} and {}: {}".format(layer.name, variable_names[i], e))
            compat_error=True
            
    if compat_error:
        logging.warning("There are some layers for which weights were not successfully loaded.")
        
        
        
def create_inception_v3(input_shape,
                       n_classes,
                       model_path,
                       variable_path,
                       eval_mode,
                       freeze_hidden_layers,
                       learning_rate,
                       num_epochs_per_decay,
                       learning_rate_decay_factor,
                       rmsprop_momentum,
                       rmsprop_epsilon,
                       rmsprop_decay):
    # load weights
    try:
        # load model from file
        model = tf.keras.models.load_model(model_path)
        
        # if loaded model has another number of output neurons than n_classes: Chop off output layer and replace with n_classes neurons
        if model.layers[-1].weights[-1].shape != n_classes:
            model = tf.keras.Model(inputs=model.input, outputs=[tf.keras.layers.Dense(n_classes, name="predictions", activation="softmax")(model.layers[-2].output)])

    except:
        logging.warning("Loading model with tf.keras.models.load_model(..) failed. Restoring weights layerwise instead.")

        model=tf.keras.applications.InceptionV3(
                    include_top=True, 
                    weights=None, 
                    input_shape=input_shape, 
                    classes=n_classes
                )

        load_weights_layerwise(model, 
                           model_path,
                           variable_path,
                           output_layer=True,
                           eval_mode=eval_mode)

    # freezing hidden layers if specified
    if freeze_hidden_layers:
        model.trainable = False
        model.layers[-1].trainable = True

    # define a schedule for learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                decay_steps=num_epochs_per_decay,
                                                                decay_rate=learning_rate_decay_factor,
                                                                staircase=True)

    # define RMS prop optimizer
    opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, 
                                  momentum=rmsprop_momentum, 
                                  epsilon=rmsprop_epsilon, 
                                  decay=rmsprop_decay)

    # compile model with categorical crossentropy
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
          optimizer=opt,
          metrics=['accuracy'])

    return model


def init_callbacks(checkpoint_dir,
                  checkpoint_filename,
                  logs_dir,
                  monitor="val_accuracy",
                  es_patience=10,
                  verbose=1,
                  checkpoint=True,
                  tensorboard=True,
                  early_stopping=True):
    callbacks = []
    if checkpoint:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + '/' + checkpoint_filename,
            save_best_only=True,
            save_weights_only=True,
            monitor=monitor,
            save_freq='epoch',
            mode='auto',
            verbose=verbose))
    if tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir))
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=es_patience,
            verbose=verbose,
            mode="auto",
            restore_best_weights=True))
    return callbacks
    
    