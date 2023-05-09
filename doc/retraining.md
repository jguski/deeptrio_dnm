# retraining

|Script:|../scripts/retrain_model.py|
|:--:|:--:|
|Environment:|../envs/tensorflow.yml|

The retraining script can be used to retrain a DeepTrio model from any starting checkpoint. Since the original DeepTrio uses an earlier version of TensorFlow, if the original child model is used, its weights are extracted manually and copied to a keras instance of Inception v3. Checkpoints created by this script can be used directly.

## Input requirements

Our script expects the examples to be in the following folder pattern:
`${base}/${family}/${base}.tfrecord-?????-of-?????.gz`. The script uses this pattern to determine a suitable training / validation split based on the family information.

## Training a model

To retrain the model, you can use the following command. Required inputs are:

 * `train_data_path`: The base input folder with training examples
 * `n_classes`: 4 if retraining for DNMs only, or 5 if retraining for DNMs and CDNMs. `class_weights` are chosen accordingly, giving heigher weight to the DNM and CDNM classes: `[1, 1, 1, 10, 10]`.
 * `model_path`: Path to the starting model.

```bash
python3 retrain_model.py \
    --train_data_path ${input_dir} \
    --log_dir $TMPDIR \
    --checkpoint_dir ${output_model} \
    --epochs 100 \
    --batch_size 64 \
    --num_epochs_per_decay 15.0 \
    --learning_rate_decay_factor 0.995 \
    --learning_rate 0.001 \
    --n_classes ${n_classes} \
    --class_weights ${class_weights} \
    --variable_path scripts/models/variables.txt \
    --model_path ${model_path}
```