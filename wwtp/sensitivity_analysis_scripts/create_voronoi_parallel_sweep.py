#!/usr/bin/env python
"""
Parallel parameter sweep executor for create_voronoi.

Runs 4 instances of create_voronoi.py in parallel, each handling a subset
of parameter combinations. Designed to be called from SLURM array jobs.

Usage:
    python -m wwtp.sensitivity_analysis_scripts.create_voronoi_parallel_sweep --task-id <id> [--version <v>] [--dynamic-buffering <true|false>] [--size-measure <reported|footprint>] [--approach APPROACH]

Parameters:
    TASK_ID: SLURM array task ID (0-9)
    VERSION: Optional config version
    DYNAMIC_BUFFERING: Backward-compatible positional arg (ignored; sweep grid controls this)
    SIZE_MEASURE: Backward-compatible positional arg (ignored; sweep grid controls this)
    APPROACH: Approach to run (default: 1)
"""

import sys
import os
import logging
import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from ..starter import load_config
from ..utils import configure_logging
from ..pipelines import create_output_paths

# Setup logging
def setup_logging(log_dir: str, task_id: int) -> logging.Logger:
    """Configure stdout-only logging; the shell wrapper owns the log file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)

    return logger


def generate_parameter_combinations(weight_func_filter: str = "all", cfg=None) -> List[Tuple[int, str, int, str, str, str, str]]:
    """Generate all parameter combinations for the sweep.

    The grid comes from ``create_voronoi_parallel_sweep`` in config.yaml
    (``sweep_levels``, ``sweep_weight_methods``, ``sweep_rigid_buffers``,
    ``sweep_size_measures``); ``cfg`` is that resolved section and is loaded
    when not given.

    Combination tuple:
        (level, version, buffer, weight_method, weight_func,
         dynamic_buffering, size_measure)
    """
    if cfg is None:
        cfg = load_config(script_name="create_voronoi_parallel_sweep")
    levels = list(cfg["sweep_levels"])
    # The additive weighting stays available to single runs (and through the
    # explicit "add" filter) but is not part of the default sweep grid.
    if weight_func_filter == "all":
        weight_funcs = ["mult", ""]
    elif weight_func_filter == "add":
        weight_funcs = ["add"]
    elif weight_func_filter == "mult":
        weight_funcs = ["mult"]
    elif weight_func_filter == "none":
        weight_funcs = [""]
    else:
        raise ValueError("weight_func_filter must be one of: all, add, mult, none")
    weight_methods = list(cfg["sweep_weight_methods"])
    rigid_buffers = list(cfg["sweep_rigid_buffers"])
    size_measures = list(cfg["sweep_size_measures"])
    
    combinations = []
    for level in levels:
        for weight_func in weight_funcs:
            for weight_method in weight_methods:
                # When weight_func is empty the distance weighting is disabled,
                # so weight_method has no effect on the output. Only include
                # one canonical method (linear) to avoid redundant runs.
                if weight_func == "" and weight_method != "linear":
                    continue
                # 'size' weights only exist for the multiplicative distance.
                if weight_method == "size" and weight_func != "mult":
                    continue

                # (a) Rigid buffering regime.
                for buffer in rigid_buffers:
                    combinations.append((
                        level,
                        "",
                        buffer,
                        weight_method,
                        weight_func,
                        "false",
                        "",
                    ))

                # (b) Size-based service radius, one run per size measure.
                for q in size_measures:
                    combinations.append((
                        level,
                        "",
                        9000,
                        weight_method,
                        weight_func,
                        "true",
                        str(q),
                    ))
    
    return combinations


def filter_combinations_by_task(
    combinations: List[Tuple[int, str, int, str, str, str, str]],
    task_id: int,
    num_tasks: int = 10,
    shuffle_seed: int = 42,
) -> List[Tuple[int, str, int, str, str, str, str]]:
    """Deterministically shuffle then assign combinations to this task ID."""
    shuffled = list(combinations)
    random.Random(shuffle_seed).shuffle(shuffled)
    return [combo for idx, combo in enumerate(shuffled) if idx % num_tasks == task_id]


def print_task_combinations(
    task_id: int,
    shuffle_seed: int,
    weight_func_filter: str = "all",
    num_tasks: int = 10,
) -> None:
    """Print this task's assigned combinations as tab-separated lines.

    Emits ``level``, ``buffer``, ``weight_method``, ``weight_func``,
    ``dynamic_buffering``, ``size_measure`` per line (the ``version``
    field of the combination tuple is dropped - the sweep .sh wrappers
    supply their own version via config/CLI override). Empty strings are
    encoded as ``__EMPTY__`` since the .sh wrappers parse this with a
    tab-delimited ``read`` that a genuinely blank field would confuse.

    This is the single source of truth for the sweep parameter grid used
    by the sweep .sh wrapper scripts (create_voronoi_param_sweep.sh,
    add_pop_param_sweep.sh, industrial_analysis_sweep.sh); they call this
    via ``python -c`` instead of each reimplementing the grid.
    """
    combinations = generate_parameter_combinations(weight_func_filter)
    assigned = filter_combinations_by_task(
        combinations, task_id, num_tasks=num_tasks, shuffle_seed=shuffle_seed
    )
    for level, _version, buffer, weight_method, weight_func, dynamic_buffering, size_measure in assigned:
        wf = weight_func if weight_func != "" else "__EMPTY__"
        dbk = size_measure if size_measure != "" else "__EMPTY__"
        print(f"{level}\t{buffer}\t{weight_method}\t{wf}\t{dynamic_buffering}\t{dbk}")


def split_combinations_into_jobs(
    combinations: List[Tuple[int, str, int, str, str, str, str]],
    num_jobs: int = 4
) -> List[List[Tuple[int, str, int, str, str, str, str]]]:
    """Split combinations into num_jobs roughly equal groups."""
    if not combinations:
        return [[] for _ in range(num_jobs)]
    
    jobs = [[] for _ in range(num_jobs)]
    for idx, combo in enumerate(combinations):
        jobs[idx % num_jobs].append(combo)
    
    return jobs


def run_voronoi_job(
    job_id: int,
    combinations: List[Tuple[int, str, int, str, str, str, str]],
    approach: str,
    logger: logging.Logger,
    project_root: str,
    version: str = "",
    dynamic_buffering: str = "",
    size_measure: str = "",
    max_retries: int = 2,
) -> List[Tuple[int, str, int, str, str, str, str]]:
    """
    Worker function: Run a subset of parameter combinations.
    
    Parameters:
        job_id: Identifier for this job (0-3)
        combinations: List of
            (level, version, buffer, weight_method, weight_func,
             dynamic_buffering, size_measure) tuples
        approach: Approach ID to run
        logger: Logger instance
        project_root: Root project directory
        version: Optional version override
        max_retries: Number of retries per failed run

    Returns:
        List of parameter combinations that failed after retries
    """
    sys.path.insert(0, project_root)

    log_msg = f"Job {job_id}: Starting with {len(combinations)} parameter combinations"
    logger.info(f"[Job {job_id}] {log_msg}")

    failed_combinations: List[Tuple[int, str, int, str, str, str, str]] = []
    output_exists_cache: dict[Tuple[int, str, int, str, str, str, str], bool] = {}

    def output_exists_for_combo(combo: Tuple[int, str, int, str, str, str, str]) -> bool:
        if combo in output_exists_cache:
            return output_exists_cache[combo]

        level, combo_version, buffer, weight_method, weight_func, combo_dynamic_buffering, combo_size_measure = combo
        version_override = version or combo_version or None

        cfg = load_config(
            script_name="create_voronoi_parallel_sweep",
            level=str(level),
            version=version_override,
            buffer=int(buffer),
            weight_method=weight_method,
            weight_func=weight_func,
            dynamic_buffering=combo_dynamic_buffering if combo_dynamic_buffering else None,
            size_measure=combo_size_measure if combo_size_measure else None,
        )
        output_path = create_output_paths(cfg)["voronoi"][approach]
        exists = os.path.exists(output_path)
        output_exists_cache[combo] = exists
        return exists

    try:
        import subprocess

        for run_idx, (level, _, buffer, weight_method, weight_func, combo_dynamic_buffering, combo_size_measure) in enumerate(combinations, 1):
            log_msg = f"Job {job_id}: Run {run_idx}/{len(combinations)}: " \
                     f"level={level} buffer={buffer} weight_method={weight_method} " \
                     f"weight_func='{weight_func}' dynamic_buffering={combo_dynamic_buffering} size_measure={combo_size_measure}"
            logger.debug(f"[Job {job_id}] {log_msg}")

            # Build command
            # Only append optional flags that carry a value. An unset override
            # is None/"" here, and subprocess.run raises TypeError on a None
            # argv entry, which the broad except below would report as an
            # opaque per-job exception rather than a bad command.
            cmd = [
                sys.executable,
                "-m",
                "wwtp.create_voronoi",
                "--level",
                str(level),
                "--buffer",
                str(buffer),
                "--weight-method",
                weight_method,
                "--weight-func",
                weight_func,
                "--approach",
                approach,
            ]
            if version:
                cmd.extend(["--version", version])
            if combo_dynamic_buffering:
                cmd.extend(["--dynamic-buffering", combo_dynamic_buffering])
            if combo_size_measure:
                cmd.extend(["--size-measure", combo_size_measure])
            env = os.environ.copy()
            env["WWTP_SERVICE_PIPELINE_LOG_LEVEL"] = "INFO"

            attempt = 0
            run_succeeded = False
            while attempt <= max_retries and not run_succeeded:
                result = subprocess.run(cmd, capture_output=False, text=True, env=env)
                if result.returncode == 0:
                    run_succeeded = True
                    log_msg = f"Job {job_id}: Run {run_idx} completed successfully"
                    logger.debug(f"[Job {job_id}] {log_msg}")
                    break

                attempt += 1
                if attempt <= max_retries:
                    backoff_seconds = min(60, 5 * (2 ** (attempt - 1)))
                    logger.warning(
                        "[Job %s] Run %s failed with return code %s; retrying in %ss "
                        "(%s/%s)",
                        job_id,
                        run_idx,
                        result.returncode,
                        backoff_seconds,
                        attempt,
                        max_retries,
                    )
                    time.sleep(backoff_seconds)

            if not run_succeeded:
                combo = (level, "", buffer, weight_method, weight_func, combo_dynamic_buffering, combo_size_measure)
                if output_exists_for_combo(combo):
                    logger.warning(
                        "[Job %s] Run %s returned non-zero but output file exists; skipping retry mark",
                        job_id,
                        run_idx,
                    )
                else:
                    failed_combinations.append(combo)
                    log_msg = (
                        f"Job {job_id}: Run {run_idx} FAILED after {max_retries + 1} attempts"
                    )
                    logger.error(f"[Job {job_id}] {log_msg}")

        log_msg = f"Job {job_id}: Completed all {len(combinations)} combinations"
        logger.info(f"[Job {job_id}] {log_msg}")

    except Exception as e:
        log_msg = f"Job {job_id}: EXCEPTION: {str(e)}"
        logger.error(f"[Job {job_id}] {log_msg}")
        for combo in combinations:
            if output_exists_for_combo(combo):
                logger.warning(
                    "[Job %s] Exception path: output exists for combo %s; skipping retry mark",
                    job_id,
                    combo,
                )
                continue
            failed_combinations.append(combo)

    return failed_combinations


def execute_with_job_count(
    combinations: List[Tuple[int, str, int, str, str, str, str]],
    num_jobs: int,
    approach: str,
    logger: logging.Logger,
    project_root: str,
    version: str,
    dynamic_buffering: str,
    size_measure: str,
    retry_failed_runs: int,
) -> List[Tuple[int, str, int, str, str, str, str]]:
    """Execute a batch with a fixed number of jobs and return failed combinations."""
    job_combinations = split_combinations_into_jobs(combinations, num_jobs=num_jobs)
    non_empty_jobs = [
        (job_id, combos)
        for job_id, combos in enumerate(job_combinations)
        if combos
    ]
    for job_id, combos in enumerate(job_combinations):
        logger.info(f"Job {job_id}: {len(combos)} combinations")

    if not non_empty_jobs:
        return []

    failed_combinations: List[Tuple[int, str, int, str, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=num_jobs, thread_name_prefix="VoronoiJob") as executor:
        futures = {
            executor.submit(
                run_voronoi_job,
                job_id,
                combos,
                approach,
                logger,
                project_root,
                version,
                dynamic_buffering,
                size_measure,
                retry_failed_runs,
            ): job_id
            for job_id, combos in non_empty_jobs
        }

        for future in as_completed(futures):
            job_id = futures[future]
            try:
                failed = future.result()
                failed_combinations.extend(failed)
                logger.info(f"Job {job_id} completed (failed runs: {len(failed)})")
            except Exception as exc:
                logger.error(f"Job {job_id} failed with exception: {exc}", exc_info=True)
                failed_combinations.extend(job_combinations[job_id])

    return failed_combinations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel parameter sweep executor for create_voronoi"
    )
    parser.add_argument("--task-id", type=int, default=None,
                       help="SLURM array task ID (0-9)")
    parser.add_argument("--version", default="",
                       help="Optional config version")
    parser.add_argument("--dynamic-buffering", default="",
                       help="Optional global dynamic buffering override")
    parser.add_argument("--size-measure", default="",
                       help="Optional global service-radius quantile override")
    parser.add_argument("--approach", type=str, default="1",
                       help="Approach to run (default: 1)")
    parser.add_argument("--num-jobs", type=int, default=4,
                       help="Number of parallel jobs (default: 4)")
    parser.add_argument("--retry-failed-runs", type=int, default=2,
                       help="Retries per failed parameter run (default: 2)")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                       help="Seed for deterministic random assignment across 10 tasks")
    parser.add_argument("--weight-func-filter", type=str, default="all",
                       choices=["all", "add", "mult", "none"],
                       help="Optional weight_func subset for the sweep grid; 'all' = mult + unweighted, 'add' is opt-in")
    
    args = parser.parse_args()
    
    # Get task ID from environment or argument
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    
    if not (0 <= task_id <= 9):
        print(f"ERROR: TASK_ID must be between 0 and 9 (got {task_id})", file=sys.stderr)
        sys.exit(1)
    
    # Setup paths and logging
    # Anchored to this file, not the working directory: the sweep writes logs and
    # runs an editable install against it, and both must resolve the same way no
    # matter where the job was launched from.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    logger = setup_logging(log_dir, task_id)
    
    logger.info(f"Starting parallel Voronoi sweep (task {task_id}/9, approach={args.approach})")
    logger.info(f"Initial parallel jobs: {args.num_jobs}")
    logger.info(f"Shuffle seed: {args.shuffle_seed}")
    logger.info(f"Weight function filter: {args.weight_func_filter}")
    if args.dynamic_buffering or args.size_measure:
        logger.info(
            "Dynamic buffering global overrides were provided; "
            "sweep combinations still define rigid and dynamic modes explicitly."
        )
    
    try:
        # Install editable package
        import subprocess
        logger.debug("Installing the wwtp package (editable)")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", project_root],
            capture_output=True,
            check=False
        )
        
        # Generate and filter combinations
        all_combinations = generate_parameter_combinations(weight_func_filter=args.weight_func_filter)
        task_combinations = filter_combinations_by_task(
            all_combinations,
            task_id,
            num_tasks=10,
            shuffle_seed=args.shuffle_seed,
        )
        logger.info(f"Task {task_id}: {len(task_combinations)} combinations to process")

        # Start from num-jobs=4 (default) and progressively reduce to 1 if failures persist.
        current_jobs = max(1, args.num_jobs)
        remaining = list(task_combinations)
        while remaining and current_jobs >= 1:
            logger.info(
                "Executing %s combinations with num-jobs=%s",
                len(remaining),
                current_jobs,
            )
            failed = execute_with_job_count(
                combinations=remaining,
                num_jobs=current_jobs,
                approach=args.approach,
                logger=logger,
                project_root=project_root,
                version=args.version,
                dynamic_buffering=args.dynamic_buffering,
                size_measure=args.size_measure,
                retry_failed_runs=args.retry_failed_runs,
            )

            if not failed:
                remaining = []
                break

            logger.warning(
                "num-jobs=%s left %s failed combinations",
                current_jobs,
                len(failed),
            )
            remaining = failed
            if current_jobs == 1:
                break
            current_jobs -= 1
            logger.warning("Reducing num-jobs and retrying with num-jobs=%s", current_jobs)

        if remaining:
            logger.error(
                "Task %s completed with %s failed parameter runs even at num-jobs=1",
                task_id,
                len(remaining),
            )
            sys.exit(2)

        logger.info(f"Task {task_id}: All parallel jobs completed successfully")
        
    except Exception as e:
        logger.error(f"Task {task_id}: FAILED with exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    configure_logging()
    main()
