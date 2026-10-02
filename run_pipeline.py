"""Main pipeline runner for Flavor-Nutrition Pairing Project (Stages S1 to S8).

Usage:
    python run_pipeline.py [--stage STAGE_NAME] [--config PATH_TO_PARAMS]
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fnp.utils import get_project_root, load_params, setup_logger
from fnp.ingest import run_ingest
from fnp.filter import run_filter
from fnp.match import run_match
from fnp.profile import run_profile
from fnp.pairs import run_pairs
from fnp.analyze import run_analyze
from fnp.visualize import run_visualize

logger = setup_logger("fnp.pipeline", "logs/pipeline.log")


STAGES = {
    "S1": ("S1_ingest", run_ingest),
    "S2": ("S2_filter", run_filter),
    "S3": ("S3_match", run_match),
    "S4": ("S4_profile", run_profile),
    "S5": ("S5_pairs", run_pairs),
    "S6": ("S6_analyze", run_analyze),
    "S7": ("S7_visualize", run_visualize),
}


def log_run_start():
    """Record run initiation with timestamp and git commit if available."""
    root = get_project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=root
        ).decode().strip()
    except Exception:
        pass

    log_entry = f"--- Pipeline Run started at {datetime.now().isoformat()} [commit: {git_commit}] ---\n"
    with open(logs_dir / "pipeline_runs.log", "a", encoding="utf-8") as f:
        f.write(log_entry)


def main():
    parser = argparse.ArgumentParser(description="Run the Flavor-Nutrition Pairing pipeline.")
    parser.add_argument("--stage", choices=["S1", "S2", "S3", "S4", "S5", "S6", "S7", "all"], default="all",
                        help="Specify pipeline stage to execute (default: all)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom params.yaml")
    parser.add_argument("--force-match", action="store_true", help="Regenerate S3 candidates and back up the existing log")
    args = parser.parse_args()

    log_run_start()
    logger.info(f"Executing pipeline stage(s): {args.stage}")

    if args.stage == "all":
        stage_sequence = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    else:
        stage_sequence = [args.stage]

    for stage_id in stage_sequence:
        name, stage_fn = STAGES[stage_id]
        logger.info(f"========== Starting {stage_id}: {name} ==========")
        try:
            if stage_id == "S3":
                res = stage_fn(args.config, force=args.force_match)
            else:
                res = stage_fn(args.config)
            logger.info(f"Completed {stage_id} successfully. Result: {res}")
        except Exception as e:
            logger.exception(f"Error executing stage {stage_id}: {e}")
            sys.exit(1)

    logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    main()
