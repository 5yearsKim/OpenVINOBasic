#!/bin/bash

isort . --skip venv 
black . \
  --exclude '(/venv/|/\.venv/|/__pycache__/)' 
