#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_day01.py
