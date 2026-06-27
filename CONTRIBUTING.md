# Contributing to Alem

Thanks for your interest in improving *Alem*! Contributions of all kinds are
welcome — bug fixes, new RL/LLM baselines, coordination tasks, documentation,
and benchmark results.

## Ways to contribute

- **Report a bug** or request a feature via [GitHub Issues](https://github.com/alem-world/alem-env/issues).
- **Submit a result** to the [leaderboard](https://alem-world.github.io/leaderboard) — see the submission instructions there.
- **Open a pull request** for code or docs (see below).

## Development setup

```bash
git clone https://github.com/alem-world/alem-env.git
cd alem-env
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

(`pip install -e ".[dev]"` also works if you prefer plain pip.)

## Before you open a PR

CI runs lint, format, and the test suite on Python 3.11 and 3.12, so run the
same checks locally first:

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pytest alem/tests/    # tests
```

Auto-fix lint and formatting before committing:

```bash
uv run ruff check --fix .
uv run ruff format .
```

> The `baselines/`, `examples/`, and `visualise/` directories are intentionally
> excluded from the library's lint/format bar (see `pyproject.toml`). New code in
> the `alem/` package is expected to pass cleanly.

## Pull request checklist

- [ ] The change is focused and described clearly (link any related issue).
- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `pytest alem/tests/` passes; new behaviour has a test where practical.
- [ ] Public API changes are reflected in the README and docstrings.
- [ ] Environment dynamics changes are called out explicitly (these affect
      reproducibility — see Versioning below).

## Versioning & reproducibility

*Alem* follows [semantic versioning](https://semver.org/). Any change that alters
environment dynamics, observations, or scoring can change published results, so
flag it in your PR description so it can be released under an appropriate version
bump. The version lives in `pyproject.toml` and `CITATION.cff`; a matching
`vX.Y.Z` git tag triggers the PyPI publish workflow.

## Code of conduct

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
