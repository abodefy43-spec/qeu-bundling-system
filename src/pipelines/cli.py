"""CLI helpers for Phase 0 bootstrap and status output."""

from __future__ import annotations

import argparse
import json

from api.app import run_api
from pipelines.bootstrap import bootstrap_phase0
from pipelines.bundle_universe import build_bundle_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qeu-phase0")
    subcommands = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subcommands.add_parser("bootstrap", help="Create the Phase 0 manifest and ensure layout")
    bootstrap_parser.add_argument("--project-root", default=None)

    universe_parser = subcommands.add_parser(
        "materialize-bundle-universe",
        help="Build the offline bundle universe artifact for personalized bundle serving",
    )
    universe_parser.add_argument("--project-root", default=None)
    universe_parser.add_argument("--target-size", type=int, default=100_000)
    universe_parser.add_argument("--per-root-limit", type=int, default=18)
    universe_parser.add_argument("--root-limit", type=int, default=None)

    api_parser = subcommands.add_parser("api", help="Run the Phase 0 API")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        result = bootstrap_phase0(project_root=args.project_root)
        print(json.dumps({"manifest_path": str(result.manifest_path), **result.report}, indent=2))
        return 0

    if args.command == "materialize-bundle-universe":
        result = build_bundle_universe(
            project_root=args.project_root,
            target_size=int(args.target_size),
            per_root_limit=int(args.per_root_limit),
            root_limit=args.root_limit,
        )
        print(
            json.dumps(
                {
                    "artifact_path": str(result.artifact_path),
                    "report_path": str(result.report_path),
                    **result.report,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "api":
        run_api(host=str(args.host), port=int(args.port))
        return 0

    parser.print_help()
    return 1
