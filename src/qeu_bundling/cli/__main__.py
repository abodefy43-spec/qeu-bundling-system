"""Unified command line entrypoint for QEU Bundling."""

from __future__ import annotations

import argparse
import os
import socket
from qeu_bundling.config.paths import ensure_layout, get_paths, migrate_legacy_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qeu-bundling")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run pipeline flows")
    run_sub = run_parser.add_subparsers(dest="mode", required=True)
    full_parser = run_sub.add_parser("full", help="Run full pipeline (phases 0-9)")
    full_parser.add_argument("--seed", type=int, default=None, help="Optional run seed (default: 42)")
    full_parser.add_argument(
        "--eval-slice",
        action="store_true",
        help="Use fixed date slice + deterministic seed for offline evaluation runs",
    )
    quick_parser = run_sub.add_parser("quick", help="Run quick pipeline (phases 6-9)")
    quick_parser.add_argument("--seed", type=int, default=None, help="Optional run seed (default: random per run)")
    quick_parser.add_argument(
        "--eval-slice",
        action="store_true",
        help="Use deterministic seed + evaluation mode metadata for comparability",
    )
    quick_parser.add_argument(
        "--retrain-models",
        action="store_true",
        help="Force phase 07 model retraining instead of reusing existing model artifacts",
    )
    materialize_parser = run_sub.add_parser(
        "materialize-final",
        help="Materialize API final recommendations only (no phase reruns)",
    )
    materialize_parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap on users to materialize (overrides env)",
    )
    materialize_parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Sample users randomly when --max-users is set (default: sorted user IDs)",
    )
    materialize_parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for --random-sample mode",
    )

    serve_parser = sub.add_parser("serve", help="Run Flask dashboard")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5000)

    usage_parser = sub.add_parser("usage", help="Summarize usage CSV")
    usage_parser.add_argument("--csv", required=True)
    usage_parser.add_argument("--save", default="")

    sub.add_parser("review", help="Interactive review tool")
    sub.add_parser("migrate-data", help="Migrate old data layout into data/raw|processed|reference")
    sub.add_parser("evaluate-quality", help="Evaluate final output quality gates and write metrics")
    sub.add_parser("explain-data", help="Generate the data review pack from latest artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = get_paths()
    ensure_layout(paths)

    if args.command == "run":
        from qeu_bundling.runners.run_materialize_final import main as run_materialize_final
        from qeu_bundling.runners.run_new_results import main as run_quick
        from qeu_bundling.runners.run_pipeline import main as run_full

        if args.mode == "full":
            run_full(seed=args.seed, eval_slice=bool(getattr(args, "eval_slice", False)))
            return 0
        if args.mode == "materialize-final":
            return int(
                run_materialize_final(
                    max_users=getattr(args, "max_users", None),
                    random_sample=bool(getattr(args, "random_sample", False)),
                    random_seed=getattr(args, "random_seed", None),
                )
            )
        run_quick(
            seed=args.seed,
            eval_slice=bool(getattr(args, "eval_slice", False)),
            retrain_models=bool(getattr(args, "retrain_models", False)),
        )
        return 0

    if args.command == "serve":
        host = str(args.host).strip() or "127.0.0.1"
        port = int(args.port)
        host_lc = host.lower()

        if host_lc in {"127.0.0.1", "localhost"}:
            os.environ.setdefault("QEU_LOCAL_FAST_MODE", "1")
            os.environ.setdefault("QEU_DASHBOARD_DEFAULT_PERSON_COUNT", "5")

        from qeu_bundling.presentation.app import app as flask_app, prewarm_local_dashboard_defaults

        os.environ.setdefault("QEU_PROJECT_ROOT", str(paths.project_root))
        prewarm_local_dashboard_defaults()
        if host == "0.0.0.0":
            try:
                lan_ip = socket.gethostbyname(socket.gethostname())
            except OSError:
                lan_ip = ""
            if lan_ip and lan_ip != "127.0.0.1":
                print(f"LAN URL: http://{lan_ip}:{port}")
            else:
                print("LAN URL could not be resolved via socket.gethostbyname(socket.gethostname()).")
        flask_app.run(host=host, port=port, debug=False)
        return 0

    if args.command == "usage":
        from qeu_bundling.core import summarize_usage

        summarize_usage.main_from_args(csv=args.csv, save=args.save)
        return 0

    if args.command == "review":
        from qeu_bundling.core import review_tool

        review_tool.main()
        return 0

    if args.command == "migrate-data":
        moved = migrate_legacy_data(paths)
        if moved:
            print("Migrated files:")
            for line in moved:
                print(f"  - {line}")
        else:
            print("No legacy data files found to migrate.")
        return 0

    if args.command == "evaluate-quality":
        from qeu_bundling.core import evaluate_bundle_quality

        return int(evaluate_bundle_quality.main())

    if args.command == "explain-data":
        from qeu_bundling.core import data_review_pack

        return int(data_review_pack.main())

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
