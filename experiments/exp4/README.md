# Exp4: ensembles of completed runs

Notebooks: [08](../../notebooks/08_ensemble.ipynb) (Exp1 + Exp2) and [10](../../notebooks/10_ensemble_exp1_exp3.ipynb) (Exp1 + Exp3)

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
