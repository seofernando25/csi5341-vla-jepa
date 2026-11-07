# Dream-RSI search (closed)

Optional arm: an automated search over SmolVLM conditioning architectures, scored by held-out
prediction loss. It found lower loss but no successful policy.

- GPU study: 23 measured 500-step screens and six 1,500-step promotions. Best candidate n0008
  (residual adapter before the final decoder layer, backbone frozen) reached promotion loss
  0.2762 versus 0.3197 for the matched SmolVLM baseline.
- Its [longer confirmation](../confirmation/n0008-20261001/README.md) scored 0/10 development trials.
- The GPU pipeline rescaled images twice, so the ranking describes a degraded visual input
  ([recovery](../recovery/README.md)). Loss here is not LIBERO success.

[attempts.jsonl](attempts.jsonl) is the exported search journal; [figures/](figures/) holds the
screen, promotion and loss/latency plots. The search harness (`rsi/`), its dashboard and source
snapshots were removed after the study closed; they are in Git history at commit `82df05d`.
