#!/bin/bash

ENV="env"

# Add token to token.txt
QISKIT_TOKEN=$(cat token.txt)

python3.10 setup.py "$QISKIT_TOKEN"

# Check if the directory exists
if [ ! -d $ENV ]; then
  # If it doesn't exist, create it
  mkdir -p $ENV
  python3.10 -m venv $ENV
  pip install -r requirements.txt
  echo "Environment created: $ENV"
else
  echo "Environment already exists: $ENV"
fi

source $ENV/bin/activate