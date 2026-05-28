# cuspML Project Priors (single source)

_All project priors live in this one file. Auto-loaded by Claude every session._

_Edit policy: Claude maintains via chat. Say "记一下" / "save" to persist a finding._
_`python scripts/sync.py` rebuilds the beamer appendix + pushes the PDF._

_This first H1 ("cuspML Project Priors (single source)") is the priors-appendix_
_marker: deck frames come before it in the PDF, priors reference after. Don't rename it_
_without updating `priors_appendix_marker` in sync_config.yaml._

---

# 进度 (Progress) / Findings

*Active findings, milestone log, current focus. Default H1 when nothing more specific fits.*

## Current focus
*cusp-mapping stage-2 ML, real vs synthetic negatives — re-read when picking the next run.*

(starter — replace with real content)

# Methods / Pipeline

*How things are computed: stage-1 cusp ID, OMNI 1-min features, XGBoost stage 2, units, gotchas.*

# Code map

*Variable names, function signatures, data-loading patterns. See also auto-built code-map.md.*

# Data & Runs

*DMSP/POES sources, OMNI coverage, pilot spectra, run registry (runs.yaml), PBS jobs.*

# 文献 (Literature)

*Tiered theses, cite-key bridges. Per-paper distills stay in docs/lit/ (indexed).*

# Workflow / Rules

*sync.py pipeline, REPL engine, build scripts, project-specific gotchas.*
