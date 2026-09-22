# IPL hierarchical Bradley–Terry

Educational match-outcome model for the Indian Premier League.

Same *idea* as the NFL Gibbs project: team strength, a small number of match
effects, posterior predictives. Different *technique*: an identified
hierarchical logit, the same predictor at train and forecast time, and
probabilities that come from that predictor.

It is a study desk. It is not a betting model.

The long form of this README is `docs/research_note.tex` (compile to PDF).
This file is the map of the repo plus the short version of the argument.


## Why this exists

The NFL repo stacked team-week strength, the spread, EPA, WPA, CPOE and
recent scoring in one linear predictor, clamped coefficients when the chain
exploded, dropped those covariates at forecast time, and published a table
where a +46 point mean sat next to a 0.51 win probability.

This project is the repair, on IPL results:

- Binary winner, not a run margin. Chases and DLS make a single Gaussian
  margin the wrong observation model.
- Strength is a franchise mean \(\mu_i\) plus a season deviation.
  \(\sum_i \mu_i = 0\). That is the identifier.
- Covariates are only things you know before the first ball: a coarse home
  edge from city, and whether team 1 is chasing after the toss.
- Holdout season uses \(\mu_i\) only. We do not pretend we have \(\alpha_{i,2026}\)
  before that season is played.
- \(P(\text{team 1 wins}) = \operatorname{logit}^{-1}(\eta)\) on the same
  \(\eta\) that ranks teams. A large strength gap cannot print as 51%.


## Research framing

Question: does an identified hierarchical Bradley–Terry model, with only
pre-match home and chase, give coherent *next-season* IPL win probabilities
that beat a coin and a pooled BT?

That is the paper. It is not "add more features until Brier drops."

A result is: holdout Brier vs 0.25 and vs pooled BT; \(\mu\) intervals;
whether \(\beta_{\text{chase}}\) is doing work. A non-result is playoff
0–1 counts or using \(\alpha_{i,s^*}\) on an unseen season.

Design:

- Train all seasons before the last; test the last.
- Holdout uses franchise \(\mu\), not a new season effect.
- Baselines: coin, always team 1, pooled identified BT.
- Rank models by Brier and log loss. Report accuracy; do not rank by it.
- Sanity: \(\sum \mu = 0\) and \(p = \operatorname{logit}^{-1}(\eta)\) on the same \(\eta\).

Rolling origin (train through \(t\), test \(t+1\)) is the next experiment.
Closing-line comparison is allowed only as a rival forecast, never as a
column in \(\eta\).

Study guide:

```bash
cd docs && pdflatex research_note.tex && pdflatex research_note.tex
```


## Model

For match \(m\) in season \(s\), listed team 1 vs listed team 2:

\[
\eta_m = \alpha_{1,s} - \alpha_{2,s}
        + \beta_{\text{home}}\, h_m
        + \beta_{\text{chase}}\, c_m,
\qquad
y_m \sim \mathrm{Bernoulli}(\operatorname{logit}^{-1}(\eta_m))
\]

\[
\alpha_{i,s} \sim N(\mu_i, \sigma_\alpha^2),
\qquad
\mu_i \sim N(0,1)\ \text{with}\ \bar\mu = 0,
\qquad
\beta_\cdot \sim N(0, 0.5^2),
\qquad
\sigma_\alpha \sim \mathrm{HalfNormal}(0.5)
\]

Holdout season \(s^*\): replace \(\alpha_{\cdot,s^*}\) by \(\mu\). Home and
chase stay, because the city and the toss are known before the first ball.

- \(h_m \in \{-1,0,1\}\) is “team 1 at a mapped home city minus team 2”.
  Cities missing from the map (UAE, SA 2009, many dual homes) are 0.
  The prose says playoffs are 0; the shipped code does **not** zero a
  playoff in a mapped city. Treat that as a known mismatch.
- \(c_m = 1\) if team 1 is chasing.

Not in \(\eta\): betting odds, box-score piles, player lists, clamped
coefficients.


## Folder structure

```
ipl-hierarchical/
├── README.md                 this file
├── requirements.txt          runtime + test deps (unpinned ranges)
├── pytest.ini                testpaths=tests, pythonpath=src
├── .gitignore                venv, pyc, pytest cache
│
├── data/
│   ├── matches_raw.csv       Cricsheet info-file tape (committed)
│   └── DATA_SOURCES.md       attribution and rebuild command
│
├── docs/
│   ├── research_note.tex     study guide (model, split, 2026 result)
│   └── research_note.pdf     compile locally; may already be present
│
├── src/ipl_hier/             importable package (not installed by default)
│   ├── __init__.py           version = 0.1.0
│   ├── names.py              franchise aliases + city → home map
│   ├── data.py               clean, encode, design matrices, holdout split
│   ├── model.py              PyMC hierarchical BT + in-sample p
│   ├── predict.py            out-of-sample p from μ + rank table
│   ├── metrics.py            Brier, log loss
│   ├── baselines.py          coin / always-team1 / pooled BT (not wired into fit)
│   ├── market.py             decimal odds → rival p
│   └── stack.py              Platt / residual / λ-blend / superlearner
│
├── scripts/                  CLI entry points; add src/ to sys.path themselves
│   ├── build_matches.py      Cricsheet zip → data/matches_raw.csv
│   ├── fit.py                headline train / last-season holdout
│   ├── report_team.py        filter a fitted holdout to one franchise
│   └── stack_holdout.py      second-stage calibration (leaky default split)
│
├── tests/
│   ├── test_identities.py    Brier / log loss / monotone p
│   └── test_data.py          aliases, drop no-result, design shapes
│
└── reports/                  created by fit.py; not in the zip
    ├── franchise_strength.csv
    ├── holdout_<season>.csv
    └── metrics.json
```

`src/` is a package root, not a nested `ipl_hier/ipl_hier` tree. Scripts
and pytest both expect `PYTHONPATH=src` (pytest.ini already sets this).
There is no `pyproject.toml`; `pip install -e .` is not set up.

`.pytest_cache/` may appear after tests. It should not be committed
(already in `.gitignore`).


## Data flow

```
Cricsheet ipl_male_csv2.zip
        │
        ▼
scripts/build_matches.py          parse *_info.csv inside the zip
        │
        ▼
data/matches_raw.csv              one row per match, raw names
        │
        ▼
ipl_hier.data.clean_matches       aliases, drop ties/NR, y, home_edge, chase
        │
        ▼
ipl_hier.data.encode              team / season indices
        │
        ├────────────────────────► ipl_hier.data.holdout_last_season
        │                                   │
        │                          train < last year     test = last year
        ▼                                   ▼
ipl_hier.model.build_model              ipl_hier.predict.proba_from_mu
        │                                   │
        ▼                                   ▼
     PyMC NUTS                         P(team1 wins) from μ + β
        │                                   │
        └──────── scripts/fit.py ───────────┘
                         │
                         ▼
                   reports/*
```


## `src/ipl_hier/` — module notes

### `names.py`

- `ALIASES` folds Delhi Daredevils → Delhi Capitals, Kings XI → Punjab
  Kings, RCB Bangalore → Bengaluru, Rising Pune plural → singular.
- Defunct sides are **not** folded: Deccan stays Deccan, Kochi stays Kochi.
- `CITY_HOME` maps a lowercased city string to one living franchise.
  Ahmedabad is always Gujarat Titans, Hyderabad always SRH. That is a
  city lookup, not a year-aware host table.
- `canon_team()` is the only name normalizer. Everything else should go
  through it.

### `data.py`

- `load_raw` — `pandas.read_csv`.
- `clean_matches` — canonicalize four name columns; keep rows whose
  winner is team 1 or team 2; `y = 1{winner == team1}`;
  `home_edge = team1_home - team2_home`; chase from toss winner +
  decision.
- `encode` — sorted team list, seasons ordered by the integer before `/`
  so `2007/08` < `2009` < `2020/21`.
- `design_matrices` — numpy arrays `t1, t2, season, y, home, chase, city`
  plus sizes. City index is rebuilt from the frame you pass unless you
  supply one. The `--venue` path in `predict.py` is unsafe if train and
  test city maps differ (it `clip`s).
- `holdout_last_season(tape, season=None)` — default last season on the
  tape. Train is every row with season key **strictly less than** the
  holdout year, so naming `--holdout-season 2025` also drops 2026.

`MatchTape` is the dataclass the rest of the package passes around:
`frame`, `teams`, `seasons`, `team_index`, `season_index`.

### `model.py`

- `build_model(dm, include_chase=True, include_home=True, include_venue=False)`
  constructs the PyMC model. `mu` is centered. `alpha` is non-centered
  `mu[:,None] + sigma_alpha * alpha_raw` with shape
  `(n_teams, n_seasons)` — including team-seasons that never played.
- `posterior_predictive_proba` — in-sample \(p\) from posterior `alpha`
  (and \(\beta\), optional venue). This is **not** the holdout path.
- `summary_match_table` — attach mean / 5% / 95% \(p\) to the tape frame.

### `predict.py`

- `proba_from_mu` — holdout path. Uses `mu` differences, then adds
  `beta_home` and `beta_chase` if those names exist on the trace.
- `team_ranks` — posterior mean and 5–95% of `mu`, sorted descending.

### `metrics.py`

- `brier(y, p) = mean((p-y)**2)`. Coin = 0.25 on any binary tape.
- `log_loss(y, p)` with clipping. Coin = `log(2) ≈ 0.693`.

### `baselines.py`

Referee comparisons. **`fit.py` does not call this file.** Until you
wire it, `metrics.json` has no coin / pooled-BT rows.

- `coin`, `always_team1`, `empirical_rate`
- `pooled_bt_eta` — identified logit, no season layer. As shipped this
  is **one IRLS step** from \(p=0.5\), not a converged MLE.
- `score` — same dict shape as the stack reports.

### `market.py`

Rival forecast only.

- `decimal_to_p`, `two_way_normalize` (strip overround), `score_vs_market`.
- Used by `fit.py` only if you pass `--odds-team1` / `--odds-team2`
  column names that exist on the tape. Those columns are not in the
  committed CSV.

### `stack.py`

Second-stage layers on an already-fit \(p\):

- `fit_platt` / `apply_platt` — \(p' = \mathrm{expit}(a + b\,\mathrm{logit}(p))\)
- `residual_logit` — working-logit ridge on extra pre-match columns
- `blend` / `choose_lambda` — mix \(p_\mu\) with last-season \(p_\alpha\)
- `superlearner` — discrete nonnegative weights that sum to 1

Legal only when weights are fit on seasons **before** the season you
score. See `scripts/stack_holdout.py`.


## `scripts/` — what each command does

### `build_matches.py`

Parse Cricsheet IPL male CSV2 **info** files inside a zip. Ball-by-ball
files are ignored.

```bash
python scripts/build_matches.py --zip /path/to/ipl_male_csv2.zip --out data/matches_raw.csv
```

Writes one row per info file: season, date, venue, city, team1, team2,
toss, winner, outcome, method, event, match_number, gender.

### `fit.py` — headline experiment

```bash
PYTHONPATH=src python scripts/fit.py \
  --data data/matches_raw.csv \
  --out reports
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `data/matches_raw.csv` | tape |
| `--out` | `reports/` | output directory |
| `--draws` / `--tune` / `--chains` | 800 / 800 / 2 | NUTS |
| `--holdout-season` | last season in the file | e.g. `2025` |
| `--venue` | off | extra identified city intercept |
| `--odds-team1` / `--odds-team2` | empty | decimal-odds columns, rival only |

What it does:

1. Clean + encode the full tape.
2. Split train / holdout. Restrict test teams to the train map.
3. Remap `test.team_index` onto `train.team_index` so `mu[t1]` lines up.
4. Sample the hierarchical model on train.
5. In-sample \(p\) from `alpha`; holdout \(p\) from `mu`.
6. Write the three report files and print `metrics.json` plus the top
   ten \(\mu\).

It does **not** save `idata`, \(\hat R\), or baseline rows.

### `report_team.py`

League-wide fit, then a filter. Not a one-team model.

```bash
PYTHONPATH=src python scripts/report_team.py --team "Mumbai Indians" --reports reports
```

Prints that franchise’s \(\mu\) rank and holdout Brier/log loss after
flipping rows where the side was listed as team 2. Writes
`reports/team_Mumbai_Indians.csv`.

### `stack_holdout.py`

```bash
PYTHONPATH=src python scripts/stack_holdout.py --holdout reports/holdout_2026.csv
```

Fits Platt (and residual / superlearner if `home_edge` /
`team1_chasing` exist) on the **first 70% of that file** and scores the
last 30%. The docstring admits this is only honest if you later switch
the fit file to a previous season. Do not quote those Briers as 2026
confirmation.


## Data

`data/matches_raw.csv` is parsed from [Cricsheet](https://cricsheet.org/)
IPL male CSV2 info files (2007/08–2026). See `data/DATA_SOURCES.md`.

On the committed snapshot:

| | n |
|---|---|
| Raw rows | 1243 |
| Decided after clean (ties / NR dropped) | 1218 |
| Train if holdout = 2026 | 1146 |
| Holdout 2026 | 72 |
| Seasons | 19 |

`team1` / `team2` are Cricsheet listing order, not home / away and not
bat-first. \(y=1\) means “listed team 1 won.”

Name continuity: Delhi Daredevils → Delhi Capitals, Kings XI → Punjab
Kings, RCB Bangalore → Bengaluru. Defunct sides stay distinct.


## Run

Python 3.10+ is the less painful path. On 3.9, old ArviZ plus SciPy
≥ 1.13 raises `ImportError: cannot import name 'gaussian' from
scipy.signal`. Fix that with `pip install "scipy>=1.10,<1.13"` or
recreate the venv on 3.12.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/fit.py --data data/matches_raw.csv --out reports
```

Smoke fit:

```bash
PYTHONPATH=src python scripts/fit.py --draws 50 --tune 50 --chains 2
```

`requirements.txt` is ranges (`pymc>=5.0`, `arviz>=0.16`, …), not a lock
file. Two machines can sample different posteriors.


## How to read a report

Holdout Brier should beat \(0.25\) (the coin). Mean predicted \(P\)
should sit near the holdout base rate. Franchise \(\mu\) intervals that
cover 0 are “not separated from average.” If \(\beta_{\text{chase}}\) is
near 0, the toss-to-chase edge is weak in this specification — that is
a result, not a bug.

In-sample accuracy will look better than holdout. Quote holdout.

### One fit that already exists (holdout 2026)

| | In sample | Holdout 2026 |
|---|---|---|
| n | 1146 | 72 |
| Accuracy | 0.595 | 0.472 |
| Brier | 0.237 | 0.263 |
| Log loss | 0.666 | 0.719 |
| Base rate \(\bar y\) | — | 0.375 |
| Mean \(p\) | — | 0.504 |
| \(\hat\beta_h\) | 0.096 | |
| \(\hat\beta_c\) | 0.353 | |
| \(\hat\sigma_\alpha\) | 0.128 | |

This spec did not beat a coin on that split. Mean \(p = 0.50\) against
a team-1 rate of 0.38 is the calibration failure. Chase was not idle;
home was small. In-sample Brier 0.237 is not the result. Details in
`docs/research_note.tex` §13.


## Tests

```bash
PYTHONPATH=src pytest -q
```

Six tests, no PyMC:

- Brier of a perfect forecast is 0; log loss of a coin is \(\log 2\);
  \(p\) increases in the strength gap.
- Alias folding; no-result rows drop; `y` / chase / home_edge on a
  two-row toy tape; design-matrix shapes.

There is no test that \(\sum\mu=0\) on a real trace, no test that the
holdout year is absent from train, and no test that playoff home is 0.


## Limits

- Two teams that never overlap a season share information only through
  \(\mu\) of other sides; expansion sides are noisy.
- Home is a city lookup, not a stadium random effect.
- No player form, no pitch report, no injury list.
- IPL squads turn over every auction. \(\mu_i\) is a franchise label,
  not a roster.
- PyMC NUTS, not a hand-rolled Gibbs. Diagnostics live on the
  InferenceData object if you save it. `fit.py` does not save it.
- `baselines.py` and the study-guide “vs pooled BT” sentence are ahead
  of the fit script.
- `stack_holdout.py` splits the official holdout 70/30.

What this repo refuses on purpose: closing line inside \(\eta\), a
Gaussian run margin, using \(\alpha_{i,s^*}\) on season \(s^*\), and
ranking models by playoff accuracy.
