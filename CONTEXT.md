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

**Test probabilities**: Five independent label probabilities for each competition test image, before thresholding. These are inputs to the ensemble, distinct from a binary Kaggle submission.

**Selected checkpoint**: The model state chosen by the run's declared validation selection rule. Validation and test probabilities attributed to that model must come from this same state.

**Resume checkpoint**: The latest recoverable training state, including optimizer and schedule progress. It can differ from the selected checkpoint.

**Head-only checkpoint**: The trained five-output classifier parameters produced while the pretrained ConvNeXt backbone remains frozen. The current checkpoint was trained on folds 1–4 and selected on fold 0, making it a shared starting point for matched fold-0 screening runs, not for other held-out folds.
