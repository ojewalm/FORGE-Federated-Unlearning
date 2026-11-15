# Baseline attribution

Reference sources for each reimplemented baseline. Where we cross-checked our
implementation against authors' code, that is noted per file.

| Baseline | Paper | Reference repo | Licence | Verdict |
| --- | --- | --- | --- | --- |
| **Halimi** | Halimi, Kadhe, Rawat, Baracaldo. *Federated Unlearning: How to Efficiently Erase a Client in FL?* UpML @ ICML 2022. arXiv:2207.05521 | [IBM/federated-unlearning](https://github.com/IBM/federated-unlearning) | Apache-2.0 | Cross-checked implementation |
| **FedMemPrune** | Wei et al. *Rethinking Federated Unlearning via the Lens of Memorization.* KDD 2026. arXiv:2605.24545 | [JWei1999/Rethinking-Federated-Unlearning-via-the-Lens-of-Memorization](https://github.com/JWei1999/Rethinking-Federated-Unlearning-via-the-Lens-of-Memorization) | MIT | Cross-checked implementation |
| **FedRecovery** | Zhang, Zhu, Zhang, Xiong, Zhou. *FedRecovery: Differentially Private Machine Unlearning for Federated Learning Frameworks.* IEEE TIFS 18 (2023) | (none public) | n/a | Reimplemented from paper |
| **FedEraser** | Liu, Ma, Yang, Wang, Liu. *FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models.* IWQoS 2021 | [Zoybzo/FedEraser-Code](https://github.com/Zoybzo/FedEraser-Code) (mirror) | **no LICENSE file** — cannot vendor | Reimplemented from paper |
| **SIFU** | Fraboni, Van Waerebeke, Scaman, Vidal, Kameni, Lorenzi. *SIFU: Sequential Informed Federated Unlearning for Efficient and Provable Client Unlearning in Federated Optimization.* AISTATS 2024, PMLR v238 | [Accenture/Labs-Federated-Learning `SIFU` branch](https://github.com/Accenture/Labs-Federated-Learning/tree/SIFU) | **no LICENSE file** on the branch — cannot vendor | Reimplemented from paper |

## Reproducibility notes

- The two vendorable references (Halimi, FedMemPrune) were used only to sanity-
  check our reimplementation on a shared toy problem; no code was copied.
- All five baselines share our training substrate: same datasets, same
  partitioning, same model factory, same FedAvg base for methods that use it.
  Only the unlearning routine (and, for history-storing methods, the choice to
  retain per-round updates) differs.
- Hyperparameters follow each paper's stated defaults where given; when a paper
  leaves a knob unspecified we grid-search on a held-out CIFAR-100 slice and
  document the chosen value in the config.

## What is *not* claimed

Our reimplementations are faithful to the algorithms described in the papers,
but they are not bit-identical to the authors' original code. If a baseline
performs worse in our numbers than in its original paper, it may be because
(i) the paper tuned per-dataset in ways we did not, (ii) our federated
substrate (client sampling schedule, non-IID partition) differs from theirs,
or (iii) our reimplementation has a subtle bug. We flag any large gap
per-table in the results write-up.
