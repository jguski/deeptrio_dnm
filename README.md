# Extending DeepTrio for sensitive detection of complex _de novo_ mutation patterns

Code repository accompanying our paper titled "Extending DeepTrio for sensitive detection of complex _de novo_ mutation patterns". Included here are the final models, used to call variants and the python scripts to re-train the DeepTrio model and call variants.

**Structure:**
  * `models/` Contains two subfolders, `DNM` and `cDNM` that contain the respective re-trained DeepTrio networks with either one or two additional output neurons. These are checkpoints that can be loaded by the Tensorflow v2 / Keras libraries. `childmodel_dt` is the original child model from DeepTrio v1.3, included here for convenience if re-training is to be started from scratch.
  * `scripts/` python scripts and programs to retrain the network and evaluate the calls generated. Includes programs to convert the network output to a VCF.
  * `envs/` conda environment files to install dependencies to run the different scripts.

Please refer to the `doc/` folder for further details on how to run the code.

## Overview

To retrain DeepTrio from scratch, the following steps are to be executed:

1. Create truth vcfs with variant annotations to serve as gold standard reference. ([create-truth-vcf](./doc/create-truth-vcf.md))
2. Generate training examples for a set of trios based on the vcfs ([make-examples](./doc/make-examples.md)).
3. Retrain the network starting from any checkpoints ([retrain](./doc/retraining.md)).
4. Call variants using the retrained network ([calling](./doc/variant-calling.md)).
5. (Optionally) Repeat 1-4 with false positive variants from the first round to increase performance. [create-truth-vcf](./doc/create-truth-vcf.md) describes how the false positive and indel variants are detected and fed into the training.
6. Evaluate the network.

The scripts that have been used are described in more detail in the relevant `doc` files. When calling variants, only the steps `2.` and `4.` need to be performed.

When the create-examples, training, evaluation loop is to be run with 5 classes, including the more complex CDNM class augmented training data can be created. [data-augmentation](./doc/data-augmentation.md) details the script used to generated synthetic CDNMs with accompanying read- and vcf data.