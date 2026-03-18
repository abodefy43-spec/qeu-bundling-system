"""Minimal Phase 0 API for engine discovery and stub execution."""

from __future__ import annotations

from flask import Flask, jsonify, request

from data.paths import get_project_paths
from shared.contracts import EngineRequest
from shared.registry import execute_engine, get_engine_descriptor, list_engine_descriptors


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        paths = get_project_paths()
        return jsonify(
            {
                "name": "qeu-bundling-system",
                "phase": "phase0_cleanup",
                "status": "ready_for_engine_implementation",
                "data_root": str(paths.data_dir),
                "engines": [descriptor.name for descriptor in list_engine_descriptors()],
            }
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "phase": "phase0_cleanup"})

    @app.get("/engines")
    def engines():
        return jsonify({"engines": [descriptor.as_dict() for descriptor in list_engine_descriptors()]})

    @app.post("/engines/<engine_name>/recommendations")
    def recommendations(engine_name: str):
        descriptor = get_engine_descriptor(engine_name)
        if descriptor is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Unknown engine: {engine_name}",
                        "available_engines": [item.name for item in list_engine_descriptors()],
                    }
                ),
                404,
            )

        payload = request.get_json(silent=True) or {}
        response = execute_engine(engine_name, EngineRequest.from_payload(payload))
        status_code = {
            "invalid_request": 400,
            "not_found": 404,
            "not_ready": 503,
        }.get(response.status, 200)
        return jsonify(response.as_dict()), status_code

    return app


def run_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    app = create_app()
    app.run(host=host, port=port, debug=False)
