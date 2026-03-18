"""Registry for the three active engine stubs."""

from __future__ import annotations

from engines.bundles.engine import PersonalizedBundlesEngine
from engines.compatible.engine import CompatibleProductsEngine
from engines.fbt.engine import FrequentlyBoughtTogetherEngine
from shared.contracts import EngineDescriptor, EngineRequest, EngineResponse

_ENGINE_REGISTRY = {
    "compatible_products": CompatibleProductsEngine(),
    "frequently_bought_together": FrequentlyBoughtTogetherEngine(),
    "personalized_bundles": PersonalizedBundlesEngine(),
}


def list_engine_descriptors() -> list[EngineDescriptor]:
    return [engine.descriptor for engine in _ENGINE_REGISTRY.values()]


def get_engine_descriptor(name: str) -> EngineDescriptor | None:
    engine = _ENGINE_REGISTRY.get(str(name).strip())
    return None if engine is None else engine.descriptor


def execute_engine(name: str, request: EngineRequest) -> EngineResponse:
    engine = _ENGINE_REGISTRY[str(name).strip()]
    return engine.recommend(request)
