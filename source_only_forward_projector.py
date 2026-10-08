"""Compatibility import for the original source-only forward-projector API."""

from cbct_backprojector.source_only_forward_projector import (
    DifferentiableSourceOnlyForwardProjector,
    project,
)

__all__ = ["project", "DifferentiableSourceOnlyForwardProjector"]
