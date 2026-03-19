# Contributing

## Setup

```bash
git clone https://github.com/TATP-233/GaussainRenderer.git
cd GaussainRenderer
uv pip install -e ".[dev]"
```

## Workflow

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run checks before committing:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy gaussian_renderer && uv run pytest
```

4. Open a pull request against `main`

## Code style

- Formatter: `ruff format` (line length 120)
- Linter: `ruff check` (E, F, I rules)
- Type checker: `mypy` (strict import checking disabled)

Fix issues with:
```bash
uv run ruff check --fix .
uv run ruff format .
```

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat:     new feature
fix:      bug fix
docs:     documentation only
style:    formatting, no logic change
refactor: code restructure, no behavior change
test:     add or update tests
chore:    build, CI, dependencies
```

## Adding dependencies

- Runtime deps go in `[project.dependencies]` (keep minimal)
- Optional feature deps go in a named extra under `[project.optional-dependencies]`
- Dev/test deps go in the `dev` extra

## Tests

Tests live in `tests/`. Run with:

```bash
uv run pytest          # all tests
uv run pytest -v -k foo  # filter by name
```

- No CUDA required — mock `torch.Tensor.cuda` and `gsplat` for unit tests
- Use `BANANA_PLY` from `conftest.py` as the standard fixture file
- Mark tests that need the fixture with `@pytest.mark.skipif(not BANANA_PLY.exists(), ...)`
