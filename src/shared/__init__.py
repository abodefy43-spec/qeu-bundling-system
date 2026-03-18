"""Shared contracts for engines and pipelines."""

from .contracts import EngineDescriptor, EngineRequest, EngineResponse, RecommendationCandidate

__all__ = [
    "EngineDescriptor",
    "EngineRequest",
    "EngineResponse",
    "RecommendationCandidate",
]
