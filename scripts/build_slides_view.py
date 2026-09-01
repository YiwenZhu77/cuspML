#!/usr/bin/env python
"""Shim → research-project-kit. The real, config-driven logic lives in the kit
(shared across all projects); it reads THIS project's ./sync_config.yaml. Edit the
kit, not this file.

Dispatches by filename: a project file named scripts/<name>.py runs the kit's
<kit>/scripts/<name>.py against this project. So the same body works as sync.py,
build_code_map.py, build_slides_view.py, push_pdf_to_drive.py, etc.

    python scripts/sync.py
"""
import os
import pathlib
import runpy
import sys

PROJ = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("RESEARCH_KIT_ROOT", str(PROJ))
KIT = pathlib.Path.home() / ".claude/plugins/marketplaces/research-project-kit/scripts"
sys.path.insert(0, str(KIT))
runpy.run_path(str(KIT / pathlib.Path(__file__).name), run_name="__main__")
