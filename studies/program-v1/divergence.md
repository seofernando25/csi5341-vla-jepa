# Does offline action error predict task success?

Question: across compressed variants, does the mean physical arm-action change against B16 on the
80 held-out screen frames predict LIBERO success? Record: `divergence.json`; figure:
`report/figures/divergence.pdf`; rebuild: `vla-report-plots`. Gates from `protocol.json`.

Result, 17 precision-surface cells, smoke and development analysed separately (never pooled):

- Development (100 trials): Spearman ρ = 0.21, bootstrap 95 % interval [-0.38, 0.79], n = 17. One
  cell (language decoder 2-bit, error 3.6e-2) succeeds 20/100; all 16 cells below 1e-2 succeed
  79-86/100. Among the 16 cells passing the 77/100 gate, ρ = 0.45 [-0.15, 0.84]: a 79-86 % band is
  not an ordering.
- Smoke (10 trials): ρ = -0.49 [-0.77, -0.10], n = 17, but 14 of 17 cells sit at 8/10, so the sign
  is saturation, not ranking; among the 14 gate-passers success is constant and no correlation exists.

Reading: large error screens (the single cell above 1e-2 fails); small differences do not rank.

What it does not show: a calibrated screen threshold (one cell above 1e-2, so n = 1); any reserved
trial; any Dream-RSI node — its 24 smoke and 3 development records carry successes but no action
error, so they are excluded and counted in `divergence.json`. With n = 17 the intervals span most of
[-1, 1]: too small to conclude that the relation is monotone, or that it is absent.
