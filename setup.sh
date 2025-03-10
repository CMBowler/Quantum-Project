#!/bin/bash

ENV="env"

# Check if the directory exists
if [ ! -d $ENV ]; then
  # If it doesn't exist, create it
  mkdir -p $ENV
  python -m venv $ENV
  pip install -r requirements.txt
  echo "Environment created: $ENV"
else
  echo "Environment already exists: $ENV"
fi

source $ENV/bin/activate