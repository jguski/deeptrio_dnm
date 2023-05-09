# configuration (of network shape and default parameters for training)

network_shape = dict(
    input_shape = (300,221,6),
    n_classes = 4
)

train_params = dict(
    epochs = 10,
    batch_size = 128,
    val_split = 0.2,
    n_folds=None,
    learning_rate = 0.064,
    rmsprop_momentum = 0.9,
    rmsprop_decay = 0.9,
    rmsprop_epsilon = 1.0,
    learning_rate_decay_factor = 0.94,
    num_epochs_per_decay = 2.0
)

general_config = dict(
    log_level = 'info',
    model_path = './src/models/childmodel_dt/model.ckpt',
    variable_path = './src/models/variables.txt',
    log_dir = './logs',
    checkpoint_dir = './checkpoints',
    save_checkpoint_every = 5,
    shuffle_buffer=10000
)