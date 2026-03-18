"""Shared request and response objects for future engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RecommendationCandidate:
    product_ids: tuple[str, ...]
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "product_ids": list(self.product_ids),
            "metadata": dict(self.metadata),
        }
        if self.score is not None:
            payload["score"] = float(self.score)
        return payload


@dataclass(frozen=True)
class EngineDescriptor:
    name: str
    description: str
    required_inputs: tuple[str, ...]
    output_description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required_inputs": list(self.required_inputs),
            "output_description": self.output_description,
        }


@dataclass(frozen=True)
class EngineRequest:
    customer_id: str | None = None
    root_product_id: str | None = None
    anchor_product_ids: tuple[str, ...] = ()
    limit: int = 10
    context: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "EngineRequest":
        raw = dict(payload or {})
        customer_id = raw.get("customer_id", raw.get("user_id"))
        root_product_id = raw.get("root_product_id")
        anchor_ids = raw.get("anchor_product_ids") or ()
        if isinstance(anchor_ids, str):
            anchor_ids = [anchor_ids]
        if root_product_id is not None:
            normalized_root_product_id = str(root_product_id).strip() or None
            if normalized_root_product_id:
                anchor_ids = [normalized_root_product_id, *anchor_ids]
        else:
            normalized_root_product_id = None
        normalized_anchor_ids = tuple(str(item).strip() for item in anchor_ids if str(item).strip())
        limit = int(raw.get("count", raw.get("limit", 10)) or 10)
        context = raw.get("context") if isinstance(raw.get("context"), dict) else {}
        return cls(
            customer_id=str(customer_id).strip() or None if customer_id is not None else None,
            root_product_id=normalized_root_product_id,
            anchor_product_ids=normalized_anchor_ids,
            limit=max(1, limit),
            context=context,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "user_id": self.user_id,
            "root_product_id": self.root_product_id,
            "anchor_product_ids": list(self.anchor_product_ids),
            "limit": self.limit,
            "context": dict(self.context),
        }

    @property
    def primary_product_id(self) -> str | None:
        if self.root_product_id:
            return self.root_product_id
        if self.anchor_product_ids:
            return self.anchor_product_ids[0]
        return None

    @property
    def user_id(self) -> str | None:
        return self.customer_id


@dataclass(frozen=True)
class EngineResponse:
    engine: str
    status: str
    items: tuple[RecommendationCandidate, ...] = ()
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "status": self.status,
            "message": self.message,
            "items": [item.as_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }


def not_implemented_response(
    descriptor: EngineDescriptor,
    request: EngineRequest,
    *,
    guidance: str,
) -> EngineResponse:
    return EngineResponse(
        engine=descriptor.name,
        status="not_implemented",
        message=f"{descriptor.name} is scaffolded but not implemented yet.",
        metadata={
            "required_inputs": list(descriptor.required_inputs),
            "received_request": request.as_dict(),
            "guidance": guidance,
        },
    )
