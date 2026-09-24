Download your training data from Milon Me equipment and plot them.

By default, statistics and plots show the last year (365 days). Choose a different
range with `--range`, using `d` for days, `w` for weeks, or `y` for years
(365 days each). Use `all` to show all downloaded history:

```sh
python milon.py --range 365d
python milon.py --range 1y
python milon.py --range 4w
python milon.py --range all
```

The range only filters statistics and plots; it does not change which months
are downloaded. Older history must already be available in the local data.

Each series shows its all-time high (ATH) across the downloaded history as a
faint dashed line, with its value in the legend. A filled circle marks an ATH
within the selected range; a hollow circle marks a lower range maximum. Tied
maxima are marked at their most recent occurrence. Work records are calculated
separately for two-set and three-set training sessions.
