"""Models: shared encoder and task heads."""

from .mtl import MtlEcg, build_model

__all__ = ["MtlEcg", "build_model"]
