# cuspML — Claude Code Instructions

## Chat-Only Workflow (CRITICAL)

User does **not** open an IDE. The only project view = the compiled PDF (Mac sioyek,
auto-reload). Claude is the only writer.

Single source of truth for priors = `.claude/rules/priors.md` (H1 categories).

Every chat turn with a finding / decision / gotcha / new job / milestone / cite / unit:
1. Edit `priors.md` (route to correct H1 + H2, see tree below)
2. `python scripts/sync.py` — rebuild beamer appendix + recompile + push PDF

**HARD RULE**: ANY edit to `.claude/rules/priors.md` (or `slides/main.tex`, the bib,
`docs/*.md`) MUST be followed by `python scripts/sync.py` in the SAME turn.
Reverse triggers ("口头说一下" / "discuss only" / "先不记") → skip persist + sync.

`python scripts/sync.py` is a shim into research-project-kit; config in `./sync_config.yaml`.

## Pre-kit assets (kept as-is)

This project predates the kit. Existing artifacts stay where they are; new work routes
through the scaffold above:
- `paper/`, `papers/`, `PAPER_PLAN.md`, `AUTO_REVIEW*.md`, root `run_*.sh` — legacy, untouched.
- `slides/beamer/slides_v5.*` — legacy decks. New synced deck is `slides/main.tex`.
- `src/*.py` (loose) + `src/lib/` + `src/scripts/mvp/` + `src/kernels/cuspmap_mvp/` — keep.
  Promote loose `src/*.py` into `src/lib/<topic>.py` when reused (rebuild code-map).

## Persistence Routing

### Step 1 — pick the H1 (priors.md categories)

| H1 | Route here when |
|---|---|
| `# 进度 (Progress) / Findings` | findings, milestones, current focus — **default if nothing else fits** |
| `# Methods / Pipeline` | stage-1 cusp ID, OMNI features, XGBoost stage 2, units, computation gotchas |
| `# Code map` | variable names, function signatures, data-loading patterns |
| `# Data & Runs` | DMSP/POES/OMNI sources, pilot spectra, runs.yaml entries, PBS jobs |
| `# 文献 (Literature)` | tiered theses, cite-keys (per-paper distills stay in docs/lit) |
| `# Workflow / Rules` | sync pipeline, REPL, build scripts, project gotchas |

### Step 2 — find or create H2
Same topic + body < 80 lines → append. Else create new H2 (silent).

### Step 3 — summary line (new H2)
Right after `## H2 title`, add an italic 1-line summary: `*<keywords first> — <when to re-read>*`. ≤80 chars.

### Step 4 — sync + report
`python scripts/sync.py`, then tell user "加到 # <H1> > <H2>".

New H1 → create + report transparently. Renaming/deleting an H1 → ask first
(user's PDF mental map is anchored to H1s; the first H1 is also the appendix marker).

## sync.py (the only command)

`python scripts/sync.py` runs: build_priors_section (pandoc→beamer fragment) →
latexmk → push to enabled sinks (sync_config.yaml) → print section page index.
Don't run the sub-steps separately unless debugging.

## Auto-loaded files (`.claude/rules/`)

| File | Content | Rebuild |
|---|---|---|
| `priors.md` | single-source priors (hand, via chat) | — |
| `code-map.md` | src/lib + src/scripts index | `build_code_map.py` |
| `slides_view.md` | view of slides/main.tex | `build_slides_view.py` |
| `lit_index.md`, `lit_cite_keys.md`, `lit_summary.md` | literature indices | build_lit_* |
| `docs_index.md` | docs/ pointer | `build_docs_index.py` |

(Run as `python ~/.claude/plugins/marketplaces/research-project-kit/scripts/build_X.py`
from the project root, or wire thin shims like scripts/sync.py.)

## REPL / code structure (kit / repl.md)

- **Kernel id**: 1 kernel = 1 source combo. Naming rule: `<model>_<dataset>_<config>`
  (e.g. `xgb_realnegs_K10`, `nn_synth_dse`; existing `cuspmap_mvp` stays valid).
  Start: `bash scripts/start_kernel.sh <id>` (sets outdir + project ledger).
- **3 layers**: `src/cells.md` (explore) → `src/lib/<topic>.py` (reuse) →
  `src/scripts/<topic>/<date>_<descr>.{py,pbs}` (PBS-ready).
- Source registry: `.claude/runs.yaml`.

## Language & Code Style

- Respond in **简体中文**. Code in English. Plot text English-only.
- Keep code simple; NaN-aware NumPy; wrap logic in functions.

## End-of-chat

Send TG notification if edits were made: `bash ~/.claude/scripts/telegram.sh msg "✅ Chat finished"`.
