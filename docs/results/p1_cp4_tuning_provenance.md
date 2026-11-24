# P1 CP4 reconciliation tuning

The selected Day 4 policy uses CNN 0.05, Lomb–Scargle 0.15, PDM 0.80 and alias penalty 0.30. The Stage 2 search retained every threshold. The original and selected-weight reconciliation baselines were reproduced before selection.

## Provenance

- Checkout HEAD at application was `ef3cd02537b6832588e82789ea2d16ad753985d6`. Both reports retain their source provenance.
- Original schema-v5 configuration hash is `e30fe1647bc4018e2fd214a24ad90d97899a5e1299ae80298cb52804f28f1f3d`.
- Tuned schema-v5 configuration hash is `de1ae4a3a8f63619814f073ad0f254048e4bd2bb801c765961d82467343f1dfa`.
- [Weight search](p1_cp4_weight_search.json) is a byte-for-byte copy of the completed weight report. SHA-256 `622eee48cfac6c393db32097041c312908260f5926c7d7a3206388ea79f774dd`.
- [Stage 2 search](p1_cp4_stage2_search.json) is a byte-for-byte copy of the cached Stage 2 replay report. SHA-256 `3243183d342912fd70ae64299f0d7dabb24cc6661346854c21d47a22ee947972`.
- Both searches use the same 18 real validation and 200 synthetic development observations. The synthetic cohort has 140 known-period and 60 No Reliable Period cases. The Stage 2 report records the weight-report and cache hashes.

## Interpretation

The selected fused shortlist gives real top-1 10/18 and Recall@3 13/18, and synthetic top-1 99/140 and Recall@3 129/140. PDM-only still outperforms fused synthetic top-1 accuracy at 108/140. Synthetic No Reliable Period shortlist emissions remain 40/60. All four Stage 2 alternatives had zero objective gain and zero fifth-percentile bootstrap gain.

The finalist results are pre-verification shortlist evidence. Cohort-level end-to-end verdict performance remains unverified, including Supported coverage and false-Supported rate. The previously observed P2 verdict policy gaps and no-valid-decoy cases remain open.
