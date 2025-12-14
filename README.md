# Translator App

`translator-app` is a Qt-based overlay translator that captures screen regions, runs OCR, and routes text through language models or translation services. The repository is organized to keep UI, platform code, capture helpers, translation engines, and supporting services cleanly separated.

## Repository Layout

- `assets/` – static resources such as icons, QSS skins, and internationalized UI strings.
- `config/` – YAML-based configuration with defaults, a user-local override, and an optional schema.
- `src/translator/` – application package where UI, core logic, OCR, translation, and platform helpers live.
- `tests/` – small pytest suite covering OCR, translation helpers, diffing, and prompt templating.
- `scripts/` – helper scripts for development (`run_dev.py`) and packaging (`build.py`).
- `dist/` – ignored build artifacts produced via PyInstaller or similar.

## Getting Started

1. Create an isolated Python environment (Hatch/Hatchling or your preferred tool).
2. Install project dependencies with `pip install -e .` (or `hatch env run pip install -e .`).
3. Copy `config/default.yaml` to `config/user.yaml` and tweak the values for your machine.
4. Run `python -m translator.main` or `scripts/run_dev.py` to launch the app in development mode.

## Configuration

- `config/default.yaml` defines fallback values for the capture region, OCR provider, translation backends, hotkeys, and logging.
- `config/user.yaml` is ignored by Git and should contain machine-specific overrides.
- `config/schema.yaml` can be used with validation tooling if you introduce schema-driven config loading.

## Testing & Tooling

- Run `pytest` from the repository root to execute the lightweight tests.
- The development script equipment uses `scripts/run_dev.py` which proxies through `translator.main`.
- Update `assets/styles/*.qss` to tweak the Qt styling or drop in new icon assets.
