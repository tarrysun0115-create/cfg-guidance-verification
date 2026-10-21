"""Gaussian-location formulas of Section III-E: TV, the delta-deniability threshold w*, checks.

``wstar_p0``: scalar-channel closed forms -- TV (Eq. (13)) with sqrt(N_eff) aggregation
(Eq. (14)), the tolerance c(delta, N_eff) (Eq. (15)) and w* under the linear sensitivity model
(Eq. (38)). ``wstar_cov``: the covariance-weighted (Mahalanobis) form S_eff^2 = d^T Sigma^-1 d
(Lemma 2 / Lemma C.2) and the Ledoit-Wolf covariance estimator. ``montecarlo``: the
analytic-vs-simulated TV check of Section III-E / Appendix B. ``deniability_budget``: the rho shift
the analytic formula asks a purification attack to achieve (optional target of Section IV-E7).
"""
