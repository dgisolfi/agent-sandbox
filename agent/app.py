#!/usr/bin/env python3
"""Agent

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def _load_experiment(experiment_id: str, experiments_dir: str) -> dict:
    path = Path(experiments_dir) / f"{experiment_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Experiment not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment must be a YAML object: {path}")
    return data


def _task_from_experiment(experiment_id: str, experiments_dir: str):
    experiment = _load_experiment(experiment_id, experiments_dir)
    title = experiment.get("title", experiment_id)
    description = experiment.get("description", "")
    prompt = experiment.get("prompt", description)
    task = f"# {title}\n\n{prompt}".strip()
    config = experiment.get("agent_config") or {}
    return task, config


def _build_loop(run_id=None):
    os.getenv("SANDBOX_URL", "http://sandbox:8000")

    sandbox = SandboxClient(os.getenv("SANDBOX_URL", "http://sandbox:8000"))
    llm = OllamaClient(
        os.getenv("OLLAMA_URL", "http://ollama:11434"),
        os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b"),
    )

    for action in ["PLAN", "EXECUTE"]:
        llm.

    return


def _run_task(task, max_iterations, error_threshold, run_id=None, stop_on_complete=True, run_until_limits=False):
    loop, sandbox, llm, logger = _build_loop(run_id)

    print("[Agent] Checking sandbox connection...")
    if not sandbox.health_check():
        print("[Agent] Sandbox service is not responding")
        return 1

    print("[Agent] Checking LLM connection...")
    if not llm.health_check():
        print("[Agent] Ollama service is not responding")
        return 1

    state = {
        task:task,
        "current_goal": "Understand the task and choose the first concrete development step.",
    }
    final_state = loop.run(state)

    print(f"Status: {final_state.status}")
    print(f"Iterations: {final_state.iteration}/{final_state.max_iterations}")
    print(f"Errors: {final_state.error_count}/{final_state.error_threshold}")

    return 0 if final_state.status == "complete" else 1


def run_command(args):
    limit = args.max_iterations or int(os.getenv("MAX_ITERATIONS", "100"))
    error_threshold = args.error_threshold or int(os.getenv("ERROR_THRESHOLD", "5"))
    return _run_task(
        args.task,
        limit,
        error_threshold,
        args.run_id,
        stop_on_complete=not args.run_until_limits,
        run_until_limits=args.run_until_limits,
    )


def start_command(args):
    exp_dir = args.experiments_dir or os.getenv("EXPERIMENTS_DIR", "/experiments")
    
    task, config = _task_from_experiment(args.experiment_id, exp_dir)

    configured_limit = int(config.get("max_iterations", os.getenv("MAX_ITERATIONS", "100")))
    configured_error_threshold = int(config.get("error_threshold", os.getenv("ERROR_THRESHOLD", "5")))
    run_until_limits = bool(config.get("run_until_limits", False))

    return _run_task(
        task,
        args.max_iterations or configured_limit,
        args.error_threshold or configured_error_threshold,
        args.run_id,
        run_until_limits=run_until_limits,
    )



def build_parser():
    parser = argparse.ArgumentParser(description="Autonomous Plan/Execute/Evaluate code agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the agent on a direct task string.")
    run.add_argument("--task", required=True)


    run.add_argument("--run-id")
    run.add_argument("--run-until-limits", action="store_true")

    run.set_defaults(func=run_command)

    version = sub.add_parser("version", help="Show version information.")
    version.set_defaults(func=lambda _args: print("Autonomous Agent Controller v1.0.0") or 0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
