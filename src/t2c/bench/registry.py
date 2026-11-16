from __future__ import annotations

from .base import Benchmark
from .suites import CADPrompt, CadQueryEval, DeepCADText, SynthBench, Text2CADBench

_BENCHMARKS: dict[str, Benchmark] = {b.name: b for b in (Text2CADBench(), CadQueryEval(), CADPrompt(), DeepCADText(), SynthBench())}


def names() -> list[str]:
    return list(_BENCHMARKS)


def get(name: str) -> Benchmark:
    try:
        return _BENCHMARKS[name]
    except KeyError:
        raise KeyError(f"unknown benchmark '{name}' (available: {', '.join(_BENCHMARKS)})") from None


def resolve(requested: list[str]) -> list[Benchmark]:
    """``["all"]`` -> every prepared benchmark; otherwise the named ones."""
    if requested == ["all"]:
        return [b for b in _BENCHMARKS.values() if b.is_prepared()]
    return [get(n) for n in requested]
