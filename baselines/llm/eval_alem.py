"""
LLM agent eval script. Inspired by https://github.com/balrog-ai/BALROG/blob/main/eval.py
"""

import logging
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict

# Add parent paths
_project_root = str(Path(__file__).parent.parent.parent)
_alem_root = os.path.join(_project_root, "alem")
sys.path.insert(0, _project_root)
sys.path.insert(0, _alem_root)
sys.path.insert(0, str(Path(__file__).parent))

from eval_utils.agents import AgentFactory
from eval_utils.evaluator import EvaluatorManager
from utils import (
    collect_and_summarize_results,
    log_results_to_wandb,
    print_summary_table,
    redirect_to_file,
    save_summary_stats,
    setup_environment,
)

from alem.alem_coop.alem_state import StaticEnvParams


def _slugify_name(value: str) -> str:
    """Create a filesystem-friendly run-name fragment."""
    value = str(value).strip().replace("/", "_")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("-_.") or "run"


def _difficulty_slug(config: DictConfig) -> str | None:
    difficulty = config.get("alem", {}).get("coordination_difficulty")
    if difficulty is None:
        return None
    if OmegaConf.is_list(difficulty):
        parts = [_slugify_name(item) for item in difficulty if str(item).strip()]
        return "-".join(parts) if parts else None
    return _slugify_name(difficulty)


def _resolve_output_dir(config: DictConfig, now: datetime) -> tuple[str, str]:
    """Return (output_dir, resolved_run_name) for this eval job."""
    if config.eval.resume_from is not None:
        output_dir = os.path.abspath(str(config.eval.resume_from))
        resolved_run_name = Path(output_dir).name
        return output_dir, resolved_run_name

    difficulty_slug = _difficulty_slug(config)
    run_name_override = config.eval.get("run_name")
    if run_name_override:
        run_name = _slugify_name(run_name_override)
    else:
        primary_model_id = config.clients[0].model_id
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        model_slug = _slugify_name(primary_model_id)
        run_name = f"{timestamp}_{config.agent.type}_{model_slug}"

    if difficulty_slug:
        run_name = f"{run_name}_{difficulty_slug}"

    output_dir = os.path.join(config.eval.output_dir, run_name)
    return output_dir, run_name


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    """Main evaluation entry point."""
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    now = datetime.now()
    output_dir, resolved_run_name = _resolve_output_dir(config, now)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "clients" not in config or len(config.clients) == 0:
        raise ValueError(
            "Missing required config.clients list. Configure one client per agent index."
        )
    primary_model_id = config.clients[0].model_id

    # Initialize W&B
    wandb_section = config.get("wandb", {})
    entity = wandb_section.get("entity") or config.get("ENTITY")
    project = wandb_section.get("project") or "alem"
    wandb_mode = config.get("WANDB_MODE", "online")

    llm_type = config.agent.type
    alg_name = f"llm_{llm_type}_{primary_model_id.replace('/', '_')}"
    default_name = f"{alg_name}_{resolved_run_name}"
    run_name_wandb = wandb_section.get("name") or default_name

    default_tags = [alg_name, "alem-coop", "multi_agent", "llm"]
    wandb_tags = wandb_section.get("tags", [])
    tags = list(set(default_tags + wandb_tags))

    group = wandb_section.get("group")
    run_id = wandb_section.get("id")
    resume = wandb_section.get("resume")
    notes = wandb_section.get("notes")

    llm_interface_version = config.get("alem", {}).get("wrapper", {}).get("version", "unknown")
    tags = list(set(tags + [StaticEnvParams.version, llm_interface_version]))

    alem_cfg = config.get("alem", {})
    wandb_config = {
        **OmegaConf.to_container(config, resolve=True),
        "ALG_NAME": alg_name,
        "env_version": StaticEnvParams.version,
        "llm_interface_version": llm_interface_version,
        # Flat uppercase keys from the alem section for W&B comparability with RL runs
        "NUM_AGENTS": alem_cfg.get("num_agents"),
        "SHARED_REWARD": alem_cfg.get("shared_reward"),
        "SOFT_SPECIALIZATION": alem_cfg.get("soft_specialization"),
        "COORDINATION_DIFFICULTY": alem_cfg.get("coordination_difficulty"),
        "MAX_TIMESTEPS": alem_cfg.get("max_timesteps"),
        "GOD_MODE": alem_cfg.get("god_mode"),
    }

    run = wandb.init(
        entity=entity,
        project=project,
        tags=tags,
        config=wandb_config,
        mode=wandb_mode,
        name=run_name_wandb,
        group=group,
        id=run_id,
        resume=resume,
        notes=notes,
        save_code=True,
        settings=wandb.Settings(init_timeout=240),
    )

    # Propagate run id to evaluator code for deterministic artifact naming.
    if "wandb" not in config:
        config.wandb = OmegaConf.create({})

    with open_dict(config.wandb):
        config.wandb.run_id = run.id

    # Setup logger
    log_filename = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )
    logging.info("Output directory: %s", output_dir)
    logging.info("Resolved run name: %s", resolved_run_name)

    upload_only = config.eval.get("upload_only", False)
    if upload_only:
        logging.info(
            "upload_only=true — skipping evaluator.run(); uploading existing results from %s",
            output_dir,
        )
    else:
        # Create an EvaluatorManager and run evaluation
        evaluator_manager = EvaluatorManager(
            config, original_cwd=original_cwd, output_dir=output_dir
        )
        agent_factory = AgentFactory(config)
        try:
            with redirect_to_file(log_filename):
                evaluator_manager.run(agent_factory)
        except Exception as e:
            logging.error(f"Evaluation run failed: {e}", exc_info=True)

    # Always collect and log whatever results exist, even on partial failure
    summary = collect_and_summarize_results(output_dir)
    print_summary_table(summary)
    save_summary_stats(summary, output_dir)

    # Log results to W&B (including debug files, GIFs, HTML)
    log_results_to_wandb(summary, config, output_dir=output_dir)

    run.finish()


if __name__ == "__main__":
    main()
