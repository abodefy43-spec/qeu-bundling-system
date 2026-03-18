"""Materialize feedback telemetry feature artifacts for Stage 2 ranking."""

from __future__ import annotations

import json
from pathlib import Path

from qeu_bundling.config.paths import ensure_layout, get_paths
from qeu_bundling.core.feedback_event_etl import (
    DEFAULT_CARRYOVER_ROWS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_PAIR_ITEMS_PER_SESSION,
    DEFAULT_MAX_VIEW_ONLY_NEGATIVES_PER_SESSION,
    build_feedback_feature_artifacts,
)


def main(
    *,
    input_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    carryover_rows: int = DEFAULT_CARRYOVER_ROWS,
    max_pair_items_per_session: int = DEFAULT_MAX_PAIR_ITEMS_PER_SESSION,
    max_view_only_negatives_per_session: int = DEFAULT_MAX_VIEW_ONLY_NEGATIVES_PER_SESSION,
) -> int:
    paths = get_paths()
    ensure_layout(paths)
    resolved_input = None if input_path is None else Path(input_path).resolve()
    result = build_feedback_feature_artifacts(
        base_dir=paths.project_root,
        input_path=resolved_input,
        chunk_size=int(chunk_size),
        carryover_rows=int(carryover_rows),
        max_pair_items_per_session=int(max_pair_items_per_session),
        max_view_only_negatives_per_session=int(max_view_only_negatives_per_session),
    )
    print(
        json.dumps(
            {
                "input_path": str(result.input_path),
                "clean_events_path": str(result.clean_events_path),
                "actor_session_features_path": str(result.actor_session_features_path),
                "actor_item_features_path": str(result.actor_item_features_path),
                "session_item_features_path": str(result.session_item_features_path),
                "proxy_pair_labels_path": str(result.proxy_pair_labels_path),
                "report": result.report,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
