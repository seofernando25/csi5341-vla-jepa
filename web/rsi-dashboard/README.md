# Experiment dashboard

Browse http://127.0.0.1:8765. The installed `csi5341-rsi-dashboard` user service starts after reboot; otherwise run `python scripts/rsi_dashboard.py`.

Chart-first view with tabs for all recorded evaluation runs, search proposals and activity. Select a point or row for evidence. Proposal timelines preserve rejections, parents and outcomes; active candidates are highlighted. Updates every three seconds; localhost only, read-only, no external assets.

The Current run selector separates cloud training, the local learning-rate branches and rollouts. Local curves average two microbatches per optimizer update; native microstep counts remain visible.

Results are separated by protocol: final LIBERO success versus inference resources, all inference benchmarks with timing-protocol filters, historical SmolVLM validation, and fresh Dream-RSI screens/promotions. Search loss is not LIBERO success. Historical paper evaluation never enters search discovery. Charts use completed measurements only; no fabricated points. CPU image preprocessing may contribute to bursty GPU activity; existing logs do not provide a phase-level profile.
