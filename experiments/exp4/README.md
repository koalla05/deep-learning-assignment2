# Exp4: ensembles of completed runs

Notebooks: [08](../../notebooks/08_ensemble.ipynb) (Exp1 + Exp2), [10](../../notebooks/10_ensemble_exp1_exp3.ipynb) (Exp1 + Exp3), [13](../../notebooks/13_ensemble_exp1_exp3_exp5.ipynb) (Exp1 + Exp3 + Exp5) and [15](../../notebooks/15_stacking_logreg.ipynb) (logistic-regression stacking)

This experiment combines completed fold-0 runs without retraining. The first
ensemble is the equal-weight mean of Exp1 and Exp2 probabilities at the fixed
0.25 threshold. Weights and thresholds are not tuned: fold 0 has only 8–12
photos of each named pet.

In notebook 08 both members start from the same backbone and Exp0 head, so their
errors overlap and the mean (public 0.96824) did not beat Exp1 (0.97003). Notebook
10 averages Exp1 with Exp3, a different CNN family (EfficientNetV2-S, public
0.97015) that disagrees with Exp1 in 219 test cells, 3.3 times as many as Exp2 does. The Exp1 + Exp3 mean scored 0.97761 on the public leaderboard, our best result so far.

Each member's test probabilities are saved once under
`experiments/<exp>/results/<run-id>_test/` and reused by later ensembles. The
ensemble's fold-0 probabilities and run record go under `results/<run-id>/`;
the submission CSV goes to `submissions/`. Kaggle upload is manual.

Flip test-time augmentation on top of this mean ([notebook 11](../../notebooks/11_tta_ensemble.ipynb)) scored 0.97420 on the public leaderboard, below the plain mean (0.97761). It changed only 32 test cells, five of them named-pet labels, and we do not use it.

The three-model mean of Exp1, Exp3 and Exp5 ([notebook 13](../../notebooks/13_ensemble_exp1_exp3_exp5.ipynb)) scored 0.98047, our best public result, although Exp5 alone scored 0.94035.

Logistic-regression stacking of the same three members ([notebook 15](../../notebooks/15_stacking_logreg.ipynb)) fits one regression per label on the members' fold-0 logits and is validated by repeated 5-fold CV inside fold 0. Its out-of-fold macro F1 (0.99878) is above the probability mean's (0.98656), but the whole difference is one Biatrix photo. The learned weights are nearly equal, so the model behaves like a mean of logits. On the test set it only removes positives, 26 of them named pets. Public score: 0.96685, which is 0.0136 below the probability mean, so we keep the mean. The drop fits most of the 26 removed named-pet cells being real pets.
