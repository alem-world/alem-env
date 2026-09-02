#!/usr/bin/env python3
"""Turn eval outputs into a ready-to-submit leaderboard entry + verification bundle.

After running `scripts/run_llm_eval.sh MODEL --difficulty easy,medium,hard`, this
reads the per-episode results, computes the Base% / Coord.% / Total% scores with 95%
CIs (rliable stratified bootstrap over episodes), prints the exact JSON object to paste
into the leaderboard, and copies the debug videos + traces into a single bundle the
maintainers use to mark an entry ✓ verified.

Usage:
    python scripts/make_submission.py --model-id meta-llama/Llama-3.2-1B-Instruct
    python scripts/make_submission.py --model-id gpt-4o-mini --type proprietary \
        --name "GPT-4o mini" --family GPT --params "—"

Difficulty runs are auto-discovered as the most recent
`outputs/alem_eval/*_<model>_<difficulty>/` dir; override any of them with
--easy/--medium/--hard <run-dir>.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import zipfile

import numpy as np

DIFFICULTIES = ("easy", "medium", "hard")
# Leaderboard field -> per-episode key inside each run json's `user_info`.
SCORE_KEYS = {
    "base": "Team/normal_reward_pct_of_max",
    "coord": "Team/coord_reward_pct_of_max",
    "total": "Team/reward_pct_of_max",
}
BOOTSTRAP_REPS = 10000  # rliable stratified-bootstrap resamples for the 95% CI
# Seeded so a leaderboard entry is reproducible: re-running this on the same run
# directory must yield the same interval, or a reviewer cannot check a number.
# It also keeps arch >= 8 working -- rliable forwards random_state=None straight
# into StratifiedBootstrap, and arch now treats an unrecognised kwarg as data:
#   TypeError: Only NumPy arrays and pandas DataFrames and Series are supported
#              in keyword arguments. Input `random_state` has type NoneType.
BOOTSTRAP_SEED = 9999  # matches EVAL_SEED; no statistical significance


def _pretty(entry: dict) -> str:
    """indent=2 JSON but with numeric score triples kept inline, like leaderboard.json."""
    s = json.dumps(entry, indent=2, ensure_ascii=False)
    return re.sub(
        r"\[\s*([-\d.,\s]+?)\s*\]",
        lambda m: "[" + ", ".join(re.findall(r"-?\d+\.?\d*", m.group(1))) + "]",
        s,
    )


def _slug(model_id: str) -> str:
    """kebab-case id for the leaderboard, e.g. meta-llama/Llama-3.2-1B -> llama-3.2-1b."""
    tail = model_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", tail).strip("-")


def _find_run_dir(outputs_dir: str, model_id: str, difficulty: str) -> str | None:
    """Most recent outputs dir for this model + difficulty that has a summary_stats.json."""
    model_tag = model_id.replace("/", "_")
    pattern = os.path.join(outputs_dir, f"*_{model_tag}_{difficulty}")
    dirs = [d for d in glob.glob(pattern) if os.path.isfile(os.path.join(d, "summary_stats.json"))]
    # Dir names are timestamp-prefixed, so lexical sort == chronological.
    return sorted(dirs)[-1] if dirs else None


def _per_episode_scores(run_dir: str) -> dict[str, np.ndarray]:
    """Per-episode Base/Coord/Total (0-1 fractions) read from each run json's user_info."""
    task_dir = os.path.join(run_dir, "alem", "default")
    run_files = sorted(
        f
        for f in glob.glob(os.path.join(task_dir, "default_run_*.json"))
        if re.search(r"default_run_\d+\.json$", os.path.basename(f))
    )
    cols: dict[str, list] = {field: [] for field in SCORE_KEYS}
    for f in run_files:
        ui = json.load(open(f)).get("user_info", {})
        for field, key in SCORE_KEYS.items():
            cols[field].append(float(ui.get(key, 0.0)))
    return {field: np.asarray(vals, dtype=float) for field, vals in cols.items()}


def _load_rliable():
    """Import rliable, or exit with the exact fix for the common arch/pandas break."""
    try:
        from rliable import library as rly

        return rly
    except Exception as exc:  # usually arch failing to import under pandas 3.x
        sys.exit(
            "ERROR: rliable (required for the bootstrap CIs) failed to import:\n"
            f"  {exc.__class__.__name__}: {exc}\n\n"
            "This is the arch/pandas-3 incompatibility, not a bug in this script. Fix the env:\n"
            "  uv pip install 'pandas<3' 'rliable>=1.2.0'\n"
            "(arch, which rliable uses for the bootstrap, needs pandas < 3.0.)"
        )


def _bootstrap_ci(values: np.ndarray) -> list:
    """[mean, lo, hi] as leaderboard percentages, matching plotting/print_results_table.py.

    Point estimate = plain per-episode mean; 95% CI = rliable stratified bootstrap over
    episodes (identical aggregate func and reps as the paper's `_bootstrap_mean_ci`).
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)] * 100.0  # per-episode values are 0-1 fractions
    if arr.size == 0:
        return [0.0, 0.0, 0.0]
    mean = float(arr.mean())
    if arr.size == 1:  # a single episode has no spread to bootstrap
        return [round(mean, 1), round(mean, 1), round(mean, 1)]

    rly = _load_rliable()
    scores = {"metric": arr[:, None]}
    aggregate_func = lambda x: np.array([np.mean(x)])  # noqa: E731
    _, interval_estimates = rly.get_interval_estimates(
        scores,
        aggregate_func,
        reps=BOOTSTRAP_REPS,
        random_state=np.random.RandomState(BOOTSTRAP_SEED),
    )
    iv = np.asarray(interval_estimates["metric"], dtype=float)
    low, high = iv if iv.shape == (2,) else iv[:, 0]
    return [round(mean, 1), round(float(low), 1), round(float(high), 1)]


def _scores_for_run(run_dir: str) -> tuple[dict, int, str]:
    """(scores dict, num_episodes, eval_date) for one difficulty's run."""
    per_ep = _per_episode_scores(run_dir)
    scores = {field: _bootstrap_ci(per_ep[field]) for field in SCORE_KEYS}
    n = int(next(iter(per_ep.values())).size)
    # dir name: <date>_<time>_<harness>_<model>_<difficulty>
    date = os.path.basename(run_dir).split("_")[0]
    return scores, n, date


def _copy_evidence(run_dir: str, difficulty: str, bundle_dir: str) -> list[str]:
    """Copy the summary + one debug video + one debug trace into the bundle."""
    copied = []
    src_summary = os.path.join(run_dir, "summary_stats.json")
    if os.path.isfile(src_summary):
        dst = os.path.join(bundle_dir, f"{difficulty}_summary_stats.json")
        shutil.copy(src_summary, dst)
        copied.append(dst)
    task_dir = os.path.join(run_dir, "alem", "default")
    # Combined gameplay GIF and self-contained debug HTML for episode 0.
    for pattern, suffix in (
        (f"{task_dir}/*_run_00.gif", "gif"),
        (f"{task_dir}/*_run_00_*debug.html", "debug.html"),
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            dst = os.path.join(bundle_dir, f"{difficulty}_run_00.{suffix}")
            shutil.copy(hits[0], dst)
            copied.append(dst)
    return copied


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-id",
        required=True,
        help="Model id as passed to the eval (e.g. meta-llama/Llama-3.2-1B-Instruct)",
    )
    p.add_argument("--name", help="Display name (default: derived from --model-id)")
    p.add_argument("--type", choices=["open-weight", "proprietary"], default="open-weight")
    p.add_argument("--family", default="", help="Model family, e.g. Llama / GPT / Qwen")
    p.add_argument("--params", default="—", help="Parameter count label, e.g. '1B', '8B', '—'")
    p.add_argument(
        "--config",
        default="",
        help="Short human label, e.g. 'high reasoning' (default: episode count)",
    )
    p.add_argument("--harness-version", default="robust_all_v0.1")
    p.add_argument("--outputs", default="outputs/alem_eval", help="Where eval runs were written")
    p.add_argument(
        "--out", default="outputs/submissions", help="Where to write the submission bundle"
    )
    p.add_argument("--easy", help="Explicit run dir for easy (skips auto-discovery)")
    p.add_argument("--medium", help="Explicit run dir for medium")
    p.add_argument("--hard", help="Explicit run dir for hard")
    args = p.parse_args()

    scores, dates, episodes, missing, empty = {}, {}, {}, [], []
    for diff in DIFFICULTIES:
        run_dir = getattr(args, diff) or _find_run_dir(args.outputs, args.model_id, diff)
        if not run_dir:
            missing.append(diff)
            continue
        s, n, date = _scores_for_run(run_dir)
        if n == 0:
            empty.append((diff, run_dir))
            continue
        scores[diff], dates[diff], episodes[diff] = s, date, n
        print(f"  {diff:<6} <- {run_dir}  ({n} episodes)")

    if missing:
        print(
            f"\nERROR: no eval run found for difficulty: {', '.join(missing)}.",
            file=sys.stderr,
        )
        print(
            f"Run all three, e.g.:\n  scripts/run_llm_eval.sh {args.model_id} --difficulty easy,medium,hard",
            file=sys.stderr,
        )
        print(
            "...or pass --easy/--medium/--hard <run-dir> explicitly.",
            file=sys.stderr,
        )
        return 1
    if empty:
        print(
            "\nERROR: no per-episode result files (alem/default/default_run_*.json) in:",
            file=sys.stderr,
        )
        for diff, d in empty:
            print(f"  {diff}: {d}", file=sys.stderr)
        print(
            "Re-run the eval without disabling outputs (needs eval.save_images/debug on).",
            file=sys.stderr,
        )
        return 1

    ep_min = min(episodes.values())
    config = args.config or f"harness: robust_all · {ep_min} episodes/difficulty"
    # params_sort orders rows on the site; derive billions from --params, else sort last.
    m = re.search(r"([\d.]+)\s*[bB]", args.params)
    params_sort = float(m.group(1)) if m else 9999
    entry = {
        "id": _slug(args.model_id),
        "name": args.name or args.model_id.split("/")[-1],
        "config": config,
        "type": args.type,
        "family": args.family,
        "params": args.params,
        "params_sort": params_sort,
        "harness_version": args.harness_version,
        "verified": False,
        "scores": scores,
        "eval_dates": dates,
    }

    # ── Build the verification bundle ────────────────────────────────────────
    bundle_dir = os.path.join(args.out, entry["id"])
    os.makedirs(bundle_dir, exist_ok=True)
    with open(os.path.join(bundle_dir, "entry.json"), "w") as f:
        f.write(_pretty(entry) + "\n")
    for diff in DIFFICULTIES:
        run_dir = getattr(args, diff) or _find_run_dir(args.outputs, args.model_id, diff)
        _copy_evidence(run_dir, diff, bundle_dir)
    zip_path = os.path.join(args.out, f"{entry['id']}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(bundle_dir):
            for name in files:
                fp = os.path.join(root, name)
                zf.write(fp, os.path.relpath(fp, args.out))

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(
        ""
        "LEADERBOARD ENTRY  (paste into the 'homogeneous' list of the site repo's data/leaderboard.json)"
        ""
    )
    print("=" * 72)
    print(_pretty(entry))
    print("=" * 72)
    print(f"Verification bundle: {bundle_dir}/  (videos + traces + summaries)")
    print(f"Zipped for upload:   {zip_path}")
    if ep_min < 10:
        print(
            f"\n⚠  Only {ep_min} episodes/difficulty — the standard is 20 (min 10). "
            "Re-run with --episodes 20 before submitting."
        )
    print(
        "\nSubmit: open a PR on github.com/alem-world/alem-world.github.io adding the entry\n"
        f"        above to data/leaderboard.json and attach {os.path.basename(zip_path)},\n"
        "        or email it to kaleabtessera@gmail.com. The videos let us re-check and mark the\n"
        "        entry ✓ verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
