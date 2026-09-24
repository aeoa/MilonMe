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

Strength-series plots show their all-time high (ATH) across the downloaded history as a
faint dashed line, with its value in the legend. A filled circle marks an ATH
within the selected range; a hollow circle marks a lower range maximum. Tied
maxima are marked at their most recent occurrence. Work records are calculated
separately for two-set and three-set training sessions.

All training sources are downloaded and plotted on each normal run. Plot groups:

- `strength`: weights, eccentric uplift, reps, and work by device/muscle group.
- `bioage`: biological-age estimates and age gaps for five muscle groups (`bioage.png`).
- `progress`: monthly circuit progress and a device load-change heatmap (`progress.png`,
  `device_progress.png`).
- `muscles`: current muscle-group distribution and available balance pairs (`muscle_balance.png`).
- `plans`: stored concentric/eccentric settings for plans used in the selected range
  (`plan_settings.png`). All historical plans referenced by downloaded sessions
  are fetched, along with the current plan.
- `sessions`: session frequency, active/elapsed time, gaps, and plan transitions
  (`training_consistency.png`), last-versus-first-set rep changes at matching
  loads/durations (`set_changes.png`), and sparse recorded heart-rate measurements
  when available (`heart_rate.png`).

All user training caches live under `data/<user>/stats/`: premium monthly files
in `premium/`, plans in `plans/`, and other endpoint JSON files directly in
`stats/`. The biological-age and progress endpoints return rolling histories.
Downloads **merge** rows by timestamp (and circuit type/device where applicable):
new results update matching records, while older records stay in the cache.
Thus history accumulates beyond the server's rolling window as you keep running
the downloader. An empty response does not erase accumulated history. Download
regularly to avoid gaps when data falls outside the server's window.

Current biological age, muscle balance, and each plan also get dated snapshots
under `stats/snapshots/YYYY-MM-DD/` (last successful capture per UTC day).
This preserves future changes, but cannot reconstruct changes before the first
capture. Plan comparisons show stored settings, not what those settings were at
every historical session. The muscle-balance plot always shows the latest dated
snapshot, independently of `--range`. Monthly plots include the cutoff month;
other historical plots use the requested range. No results outside the range
means that plot is skipped and its previous image is removed. This applies to
all selected plot groups; unselected output files are left untouched.

Failed downloads retain the existing cache and print a warning, without blocking
other endpoints. `stats/manifest.json` records all sources from the latest attempt.
Existing `stats/insights/` caches migrate automatically, including accumulated
history and snapshots; the original directory is retained as a backup.

To replot without network access, or select any combination of plot groups:

```sh
python milon.py --offline --range all
python milon.py --offline --plots strength bioage sessions --range 1y
```

`--plots` replaces the earlier `--insights-only` switch; all groups are peers.
Without `--offline`, all downloads still refresh regardless of plot selection.
Both offline commands use the saved session only to identify the local user.
All session-derived charts use Europe/Berlin time, including daylight-saving
changes. API aggregate dates retain their UTC calendar-bucket labels.
The console training summary is printed once before plotting, regardless of
which plot groups are selected. It shares the strength plots' prepared data and
date range.

The code is organized by responsibility:

- `milon.py`: login and command-line options.
- `milon_api.py`: a single download workflow for every training source.
- `training_data.py`: JSON storage, history merging, cache migration, and the
  shared plotting context (records, devices, timezone, cutoff, output location).
- `training_plots.py`: all chart groups and their common plot dispatcher.
- `training_stats.py`: console training summary, separate from chart rendering.

Run regression checks with `python -m unittest discover -s tests`.
