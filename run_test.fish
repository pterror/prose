#!/usr/bin/env fish
# Test script that uses fish syntax (since we're in fish shell)

# Source direnv to reload environment
direnv allow

# Run the test with proper environment
.venv/bin/python scripts/test_data_pipeline.py
