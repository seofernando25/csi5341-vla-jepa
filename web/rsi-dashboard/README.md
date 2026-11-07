# Experiment dashboard

Read-only, localhost-only view of recorded evaluation runs, Dream-RSI proposals and training
activity. Run `python scripts/rsi_dashboard.py` and open http://127.0.0.1:8765.

Tabs separate results by protocol: final LIBERO success versus inference resources, inference
benchmarks (filter by timing protocol), SmolVLM validation curves, and Dream-RSI screens and
promotions. Select a point or row to see its evidence. Charts show completed measurements only and
refresh every three seconds. Search loss is not LIBERO success.
