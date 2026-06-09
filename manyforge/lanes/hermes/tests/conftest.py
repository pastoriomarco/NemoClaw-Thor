"""Pytest path bootstrap for the Hermes lane tests.

Makes the manyforge/ root importable (for ``lanes.hermes.*``, ``common.*``,
``assistant_session.*``, ``openclaw_assistant_bridge.*``) and this tests/ dir
importable (for ``_fakes``) without an install step — mirroring common/tests.
The lane core (dispatcher/observer/transport/engine) depends only on the
stdlib-backed universal core, so these tests run under base Python with no
fastapi/httpx/pydantic/prometheus installed.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_MANYFORGE_ROOT = str(_HERE.parents[3])  # tests -> hermes -> lanes -> manyforge
_TESTS_DIR = str(_HERE.parent)

for _p in (_MANYFORGE_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
