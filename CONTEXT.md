# Cats and Dogs++

Multi-label recognition of cats, dogs, and three named pets in competition images.

## Language

**Named pet**: One of Moris, Motya, or Biatrix. These labels can coexist with species labels and with other named-pet labels.

**Embedding baseline**: The existing reference predictor using frozen ConvNeXt features and a separate logistic regression for each label.

**Standard ConvNeXt benchmark**: ConvNeXt with its original backbone and pooling and a five-output task classifier, without our custom feature block. Head-only training and backbone fine-tuning are separate benchmark runs.

**Custom variant**: A model containing an architectural change we implement, compared with its standard architecture under a matched training protocol.

**Public leaderboard score**: The competition score on the publicly scored portion of the hidden test labels. It is distinct from a local validation score.

**Out-of-fold predictions**: Predictions for training images made by models whose training data excludes the respective image's validation fold.

**Experiment**: A specified model and training policy to compare against a reference under the same validation protocol. An experiment may have several runs.

**Run**: One execution of an experiment with a fixed fold, seed, configuration, and code revision. A screening run on one fold is distinct from a five-fold evaluation.

**Head-only checkpoint**: The trained five-output classifier parameters produced while the pretrained ConvNeXt backbone remains frozen. It is the common starting point for matched standard and custom ConvNeXt runs.
