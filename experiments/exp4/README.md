# Exp4: ensembles of completed runs

[Notebook](../../notebooks/08_ensemble.ipynb)

This experiment combines completed fold-0 runs without retraining. The first
ensemble is the equal-weight mean of Exp1 and Exp2 probabilities at the fixed
0.25 threshold. Weights and thresholds are not tuned: fold 0 has only 8–12
photos of each named pet.

Both members start from the same backbone and Exp0 head, so their errors are
likely correlated and a small gain is expected. A different CNN family (Exp3)
joins the ensemble once its fold-0 run is complete.

Each member's test probabilities are saved once under
`experiments/<exp>/results/<run-id>_test/` and reused by later ensembles. The
ensemble's fold-0 probabilities and run record go under `results/<run-id>/`;
the submission CSV goes to `submissions/`. Kaggle upload is manual.
