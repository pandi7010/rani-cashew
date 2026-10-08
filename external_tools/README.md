# external_tools

Unmodified copies from `/home/phoenix/Documents/RaniVision_Latest` (copied 2026-10-08).
The originals were not changed.

| File | Used for |
|---|---|
| `tcm_magic_wand.py` | magic wand polygon (annotation, auto-teach). Needs `FastSAM-s.pt` (in the project folder). |
| `tcm_studio_pro.py` | TCM Studio Pro, started from the Teaching CNN page |
| `tcm_config.py`, `tcm_engine.py` | imported by `tcm_studio_pro.py` |

`tcm_config.py` keeps its projects under `C:/TCM_VISION`. On Linux that is a relative path, so the
grader starts TCM Studio with this folder as working directory and its data stays in here.
