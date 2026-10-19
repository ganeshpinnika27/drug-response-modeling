#!/usr/bin/env bash
# Usage: ./run_all.sh [config]      (default config/config.yaml)
#        ./run_all.sh --test        (synthetic smoke test)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1      # show per-drug progress live (e.g. under mamba run)
CFG=${1:-config/config.yaml}
if [[ "$CFG" == "--test" ]]; then
  python tests/make_synthetic.py tests/synthetic
  CFG=tests/synthetic/config.yaml
fi
python src/prepare_data.py   --config "$CFG"
python src/associations.py   --config "$CFG"
python src/train_models.py   --config "$CFG"
python src/explain_and_plot.py --config "$CFG"
python src/validate_prism.py --config "$CFG"     # skipped if the config has no prism section
