#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Run install.sh first."
    exit 1
fi

source venv/bin/activate
python run.py "$@"
