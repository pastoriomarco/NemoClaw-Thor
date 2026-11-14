# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[spec-step] Per-step phase timing for speculative decoding (V2 runner).

Enabled by ``VLLM_SPEC_STEP_TIMING=<window>`` (number of decode steps per log
line); unset or ``0`` leaves ``SPEC_STEP_TIMER`` as ``None`` and every call site
is a single ``is not None`` check.

A step is the span from ``execute_model`` entry to the end of
``sample_tokens``; the cycle is the host time between two step entries (it
includes the engine-core work between steps). Call sites ``mark(name)`` the end of each phase; a phase is
the interval since the previous mark. Each mark takes a host timestamp and
records a CUDA event on the current stream, so a phase has a host time (the
CPU spent issuing it) and a GPU time (the GPU span between the two events,
idle gaps included). Events are resolved with ``Event.query()`` once the
step's last event has completed, so the timer never synchronizes.
"""

import os
import time
from collections import defaultdict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class SpecStepTimer:
    def __init__(self, window: int):
        self.window = window
        self._names: list[str] = []
        self._host: list[float] = []
        self._events: list[torch.cuda.Event] = []
        self._label: str | None = None
        self._last_begin: float | None = None
        self._cycle = 0.0
        # Completed steps whose events may still be pending on the GPU.
        self._pending: list[tuple[str, float, list[str], list[float], list]] = []
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._count = 0
        self._other_steps = 0
        self._sum_cycle = 0.0
        self._sum_host: dict[str, float] = defaultdict(float)
        self._sum_gpu: dict[str, float] = defaultdict(float)
        self._num: dict[str, int] = defaultdict(int)
        self._order: list[str] = []

    @property
    def active(self) -> bool:
        return self._label is not None

    def _record(self, name: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._names.append(name)
        self._host.append(time.perf_counter())
        self._events.append(event)

    def begin_step(self) -> None:
        now = time.perf_counter()
        self._cycle = 0.0 if self._last_begin is None else now - self._last_begin
        self._last_begin = now
        self._names, self._host, self._events = [], [], []
        self._label = "other"
        self._record("begin")

    def set_label(self, label: str) -> None:
        if self._label is not None:
            self._label = label

    def mark(self, name: str) -> None:
        if self._label is not None:
            self._record(name)

    def end_step(self) -> None:
        if self._label is None:
            return
        self._pending.append(
            (self._label, self._cycle, self._names, self._host, self._events)
        )
        self._label = None
        self._resolve()

    def _resolve(self) -> None:
        while self._pending and self._pending[0][4][-1].query():
            label, cycle, names, host, events = self._pending.pop(0)
            if label != "decode":
                self._other_steps += 1
                continue
            self._count += 1
            self._sum_cycle += cycle
            for i in range(1, len(names)):
                name = names[i]
                if name not in self._num:
                    self._order.append(name)
                self._num[name] += 1
                self._sum_host[name] += (host[i] - host[i - 1]) * 1000.0
                self._sum_gpu[name] += events[i - 1].elapsed_time(events[i])
            self._sum_host["step"] += (host[-1] - host[0]) * 1000.0
            self._sum_gpu["step"] += events[0].elapsed_time(events[-1])
            if self._count >= self.window:
                self._log()
                self._reset_stats()

    def _log(self) -> None:
        n = self._count
        cycle_ms = self._sum_cycle / n * 1000.0
        phases = " | ".join(
            f"{name} {self._sum_host[name] / self._num[name]:.2f}/"
            f"{self._sum_gpu[name] / self._num[name]:.2f}"
            + ("" if self._num[name] == n else f" (x{self._num[name] / n:.2f})")
            for name in self._order
        )
        logger.info(
            "[spec-step] %d decode steps (+%d other): cycle %.2f ms (%.2f steps/s), "
            "step %.2f/%.2f ms host/gpu | %s",
            n,
            self._other_steps,
            cycle_ms,
            1000.0 / cycle_ms if cycle_ms > 0 else 0.0,
            self._sum_host["step"] / n,
            self._sum_gpu["step"] / n,
            phases,
        )


def _make_timer() -> SpecStepTimer | None:
    window = int(os.environ.get("VLLM_SPEC_STEP_TIMING", "0") or "0")
    if window <= 0:
        return None
    logger.info("[spec-step] phase timing on, window %d decode steps", window)
    return SpecStepTimer(window)


SPEC_STEP_TIMER: SpecStepTimer | None = _make_timer()
