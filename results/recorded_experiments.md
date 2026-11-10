# Recorded experiment results (historical)

This table transcribes **aggregate, previously recorded** results from the researcher's experiment documentation. These metrics were **not** produced by running the refactored source package, and the underlying participants' EEG data are not available in this repository.

All metrics below are from the **validation checkpoint selected by the highest AUROC within each run**; they are not the last epoch's scores. The experiment scripts used a split named `test.npz` as an epoch-by-epoch selection set, so **none of these values should be described as performance on an untouched test set**.

| Learning rate | Configured epochs | Accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| 1e-5 | 10 | 0.59630 | 0.53211 | 0.54622 |
| 1e-5 | 20 | 0.61296 | 0.56519 | 0.57357 |
| 1e-5 | 25 | 0.54259 | 0.53814 | 0.58812 |
| 1e-5 | 30 | 0.54545 | 0.54523 | 0.55630 |

These values are separate recorded configurations; there is no controlled external baseline or demonstrated real-world generalization improvement. The original split construction, subject-level independence, exact meaning of label codes, and class prevalence remain unverified. In particular, higher accuracy does not necessarily indicate better minority-class recognition.
