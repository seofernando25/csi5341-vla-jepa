# Dream-RSI search (closed)

Optional arm: an automated search over SmolVLM conditioning architectures, scored by held-out
prediction loss. It found lower loss but no successful policy.

- GPU study: 23 measured 500-step screens and six 1,500-step promotions. Best candidate n0008
  (residual adapter before the final decoder layer, backbone frozen) reached promotion loss
  0.2762 versus 0.3197 for the matched SmolVLM baseline.
- Its [longer confirmation](../confirmation/n0008-20261001/README.md) scored 0/10 development trials.
- The GPU pipeline rescaled images twice, so the ranking describes a degraded visual input
  ([recovery](../recovery/README.md)). Loss here is not LIBERO success.

| Path | Contents |
|---|---|
| [attempts.jsonl](attempts.jsonl), [provenance.json](provenance.json) | Exported journal and frozen configuration/source hashes |
| [sources/](sources/) | Exact baseline and n0008 source snapshots with hashes |
| [figures/](figures/) | Screens, promotions and loss/latency Pareto plots |
| [protocol/](protocol/) | Discovery-context amendment activated on 2026-10-01 |
| [archive-pil-20260930/](archive-pil-20260930/) | Earlier CPU/PIL-processing study, superseded and kept for history |

Search code and commands: [rsi/README.md](../../rsi/README.md).
