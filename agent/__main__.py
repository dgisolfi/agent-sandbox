#!/usr/bin/env python3
"""Agent

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import logging
import os
import sys
from pathlib import Path

import click
import yaml

from agent.agent import Agent
from agent.ollama import OllamaClient
from agent.tools.sandbox import SandboxClient
from agent.utils import RunLogger


def _load_experiment(experiment_id: str, experiments_dir: str) -> dict:
    path = Path(experiments_dir) / f"{experiment_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Experiment not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment must be a YAML object: {path}")
    return data


def _load_template(experiments_dir: str) -> dict:
    path = Path(experiments_dir) / "template.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _task_from_experiment(experiment_id: str, experiments_dir: str):
    experiment = _load_experiment(experiment_id, experiments_dir)
    template = _load_template(experiments_dir)

    title = experiment.get("title", experiment_id)
    environment = template.get("environment", "").strip()
    mission = experiment.get("mission", "").strip()
    phases = experiment.get(
        "phases", experiment.get("prompt", experiment.get("description", ""))
    ).strip()

    parts = [f"# {title}"]
    if mission:
        parts.append(mission)
    if environment:
        parts.append(environment)
    if phases:
        parts.append(phases)

    config = experiment.get("agent_config") or {}
    return "\n\n".join(parts), config


def _build_loop(run_id=None):
    sandbox = SandboxClient(os.getenv("SANDBOX_URL", "http://sandbox:8000"))
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    llm = OllamaClient(
        ollama_url,
        model_name,
    )
    run_logger = RunLogger(os.getenv("RUNS_DIR", "/logs/runs"), run_id)
    runner = Agent(
        sandbox=sandbox,
        logger=run_logger,
        model_name=model_name,
        base_url=ollama_url,
        model_adapter=os.getenv("AGENT_MODEL_ADAPTER", "wrapped"),
    )
    return runner, sandbox, llm, run_logger


def _run_task(
    task,
    max_iterations,
    error_threshold,
    run_id=None,
    stop_on_complete=True,
    run_until_limits=False,
):
    loop, sandbox, llm, logger = _build_loop(run_id)

    click.echo("[Agent] Checking sandbox connection...")
    if not sandbox.health_check():
        click.echo("[Agent] Sandbox service is not responding", err=True)
        return 1

    click.echo("[Agent] Checking LLM connection...")
    if not llm.health_check():
        click.echo("[Agent] Ollama service is not responding", err=True)
        return 1

    final_state = loop.run(
        task=task,
        max_iterations=max_iterations,
        error_threshold=error_threshold,
        stop_on_complete=stop_on_complete,
        run_until_limits=run_until_limits,
    )

    click.echo("\n" + "=" * 70)
    click.echo("AGENT EXECUTION REPORT")
    click.echo("=" * 70)
    click.echo(f"Status:     {final_state.status}")
    click.echo(f"Iterations: {final_state.iteration}/{final_state.max_iterations}")
    click.echo(f"Errors:     {final_state.error_count}/{final_state.error_threshold}")
    click.echo(f"Run log:    {logger.run_dir}")
    if final_state.current_goal:
        click.echo(f"Final goal: {final_state.current_goal}")
    if final_state.final_output:
        click.echo(f"Output:     {final_state.final_output[:500]}")
    click.echo("=" * 70)

    return 0 if final_state.status == "complete" else 1


@click.group()
@click.version_option("2.0.0", prog_name="agent")
def cli():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option("--task", required=True, help="Task string to run directly.")
@click.option(
    "--max-iterations",
    type=int,
    envvar="MAX_ITERATIONS",
    default=100,
    show_default=True,
)
@click.option(
    "--error-threshold",
    type=int,
    envvar="ERROR_THRESHOLD",
    default=5,
    show_default=True,
)
@click.option("--run-id", default=None)
@click.option("--run-until-limits", is_flag=True)
def run(task, max_iterations, error_threshold, run_id, run_until_limits):
    """Run the agent on a direct task string."""
    sys.exit(
        _run_task(
            task,
            max_iterations,
            error_threshold,
            run_id,
            stop_on_complete=not run_until_limits,
            run_until_limits=run_until_limits,
        )
    )


@cli.command()
@click.argument("experiment_id")
@click.option("--max-iterations", type=int, default=None)
@click.option("--error-threshold", type=int, default=None)
@click.option(
    "--experiments-dir",
    envvar="EXPERIMENTS_DIR",
    default="./experiments",
    show_default=True,
)
@click.option("--run-id", default=None)
def start(experiment_id, max_iterations, error_threshold, experiments_dir, run_id):
    """Run the agent on an experiment YAML file."""
    try:
        task, config = _task_from_experiment(experiment_id, experiments_dir)
    except Exception as exc:
        click.echo(f"[Agent] Error: {exc}", err=True)
        sys.exit(1)
    configured_limit = int(
        config.get("max_iterations", os.getenv("MAX_ITERATIONS", 100))
    )
    configured_threshold = int(
        config.get("error_threshold", os.getenv("ERROR_THRESHOLD", 5))
    )
    run_until_limits = bool(config.get("run_until_limits", False))
    stop_on_complete = bool(config.get("stop_on_complete", not run_until_limits))
    sys.exit(
        _run_task(
            task,
            max_iterations or configured_limit,
            error_threshold or configured_threshold,
            run_id,
            stop_on_complete=stop_on_complete,
            run_until_limits=run_until_limits,
        )
    )


@cli.command("list")
@click.option(
    "--experiments-dir",
    envvar="EXPERIMENTS_DIR",
    default="./experiments",
    show_default=True,
)
def list_experiments(experiments_dir):
    """List available experiments."""
    exp_dir = Path(experiments_dir)
    if not exp_dir.exists():
        click.echo(f"Error: experiments directory not found: {exp_dir}", err=True)
        sys.exit(1)

    experiments = sorted(p for p in exp_dir.glob("*.yaml") if p.stem != "template")
    if not experiments:
        click.echo(f"No experiments found in {exp_dir}")
        return

    click.echo("\nAVAILABLE EXPERIMENTS")
    click.echo("=" * 70)
    for path in experiments:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            click.echo(f"\n{data.get('id', path.stem)}")
            click.echo(f"  Title: {data.get('title', 'No title')}")
            desc = str(data.get("description", "No description"))
            click.echo(
                f"  Description: {desc[:77] + '...' if len(desc) > 80 else desc}"
            )
        except Exception as exc:
            click.echo(f"\n{path.stem}")
            click.echo(f"  Error: {exc}")


@cli.command("show-run")
@click.argument("run_id")
def show_run(run_id):
    """Print a readable transcript for a prior run."""
    runs_dir = Path(os.getenv("RUNS_DIR", "/logs/runs"))
    transcript = runs_dir / run_id / "transcript.md"
    if not transcript.exists():
        click.echo(f"Run transcript not found: {transcript}", err=True)
        sys.exit(1)
    click.echo(transcript.read_text(encoding="utf-8"))


if __name__ == "__main__":
    cli()
