"""Main-grid (p0) sweep: orchestration, leakage-proof splits and grid-level analyses.

``grid`` enumerates the (decoder, w, QF, sigma_blur, prompt, seed) cells and assigns leakage-proof
splits (grouped by prompt). ``run_p0`` drives generate -> degrade -> dual-branch invert -> feature
store + p0_table.json (resumable per decoder; weight-gated; Section IV-A, Appendix B).
``evidence`` and ``analyze`` assemble the preregistered knee-vs-w* gate and its decision memo
(Table VIII rows 1-2; Section IV-E7). ``neff_scaling`` holds the shared row loaders / cell
groupers and the per-cell sqrt(K) test (Eq. (14)); ``pooled_neff_scaling`` defines the favorable
subset and its pooled test (Section IV-E); ``postproc_pair`` is the pair-readout core of the
post-processing analysis (Table XII(a)).
"""
