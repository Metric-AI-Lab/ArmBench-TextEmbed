#!/bin/bash

# ArmBench-TextEmbed Evaluation Wrapper
# Usage: ./evaluate --config configs/config.yml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add armbench to Python path and run evaluation
export PYTHONPATH="${SCRIPT_DIR}/armbench:${PYTHONPATH}"

python -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}/armbench')
from evaluate import main
main()
" "$@"
