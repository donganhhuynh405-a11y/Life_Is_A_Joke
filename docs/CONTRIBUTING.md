# Contributing Guide

Thank you for your interest in contributing to Life Is A Joke! This guide covers how to set up the development environment, coding standards, and the pull request process.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

---

## Code of Conduct

Be respectful, constructive, and professional. Financial software carries real risk — accuracy and safety come first.

---

## Development Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional but recommended)
- Git

### Local Environment

```bash
git clone https://github.com/your-org/Life_Is_A_Joke.git
cd Life_Is_A_Joke

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Copy configuration
cp config.yaml config.local.yaml
# Edit config.local.yaml — set environment: "paper"
```

### Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

---

## Making Changes

### Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code |
| `develop` | Integration branch |
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation updates |

```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-new-strategy
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(strategy): add RSI divergence detection
fix(risk): correct position size calculation for leveraged accounts
docs(api): update WebSocket event examples
test(backtest): add edge case for zero-volume candles
```

---

## Testing

Run the full test suite before submitting:

```bash
# Unit tests
pytest tests/ -v

# ML model tests
pytest test_ml_models.py -v
pytest test_ppo_implementation.py -v
pytest test_sentiment_bert.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

All tests must pass. New features require new tests.

---

## Pull Request Process

1. **Fork** the repository and create a branch from `develop`.
2. **Write tests** for any new functionality.
3. **Run tests** — all must pass.
4. **Update documentation** in `docs/` if the change affects public APIs or behavior.
5. **Open a PR** against the `develop` branch.
6. **Fill in the PR template** — describe what changed and why.
7. **Address review feedback** promptly.

### PR Checklist

- [ ] Tests added/updated and passing
- [ ] `config.yaml` schema updated if new config keys added
- [ ] No secrets committed
- [ ] Docstrings updated for changed functions
- [ ] `docs/` updated if public interface changed

---

## Coding Standards

### Style

- **PEP 8** enforced via `flake8` / `ruff`
- Max line length: **100 characters**
- Type hints required on all public functions
- Docstrings for all public classes and methods (Google style)

```python
async def generate_signal(self, symbol: str, ohlcv: list[list]) -> dict:
    """Generate a trading signal for the given symbol.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".
        ohlcv: List of OHLCV candles [[ts, o, h, l, c, v], ...].

    Returns:
        Signal dict with keys: action, confidence, reason, price.
    """
```

### Strategy Development

- Extend `BaseStrategy` — never bypass the interface.
- Signals must return the canonical dict format (see `docs/ARCHITECTURE.md`).
- Confidence scores must be in `[0.0, 1.0]`.
- Strategies must be stateless between calls (state goes into the DB).

### Risk & Safety

- Never place real orders in test paths.
- Guard all production order logic behind `environment == "production"` checks.
- Log every order attempt with full parameters.

### Error Handling

- Use specific exception types, not bare `except Exception`.
- Exchange connectivity errors must be retried with exponential back-off.
- Critical failures must trigger a Telegram alert before raising.

---

## Adding a New Exchange

1. Add adapter in `src/core/exchange_adapter.py`.
2. Add configuration schema in `src/config.py`.
3. Add sample config block in `config.yaml`.
4. Write integration tests in `tests/test_exchange_<name>.py`.
5. Document in `docs/API.md`.

---

## Reporting Issues

- **Bug reports**: Include Python version, OS, full traceback, and config (with secrets redacted).
- **Feature requests**: Describe the use case and expected behavior.
- **Security vulnerabilities**: See [SECURITY.md](../SECURITY.md) — do **not** open a public issue.
