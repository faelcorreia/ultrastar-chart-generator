"""Heavy models are loaded once per use and released afterwards.

A module that keeps a model (or data derived from it) in memory registers a
``release`` callback; the planner calls ``release_all`` when a song is done so
models never pile up across a batch.
"""
from typing import Callable, List

_RELEASERS: List[Callable[[], None]] = []


def register_release(fn: Callable[[], None]) -> None:
    if fn not in _RELEASERS:
        _RELEASERS.append(fn)


def release_all() -> None:
    for fn in _RELEASERS:
        fn()
