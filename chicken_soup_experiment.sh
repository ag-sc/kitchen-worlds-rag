#!/bin/bash

# Check if an index argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <IDX>"
  exit 1
fi

IDX=$1
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  echo "Running experiment with IDX=$IDX (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..."
  
  python -m rag_based_prompting.evaluation.run_experiment --exp_start_idx "$IDX" --chicken_exp --plan_exp
  
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully!"
    exit 0
  else
    echo "Experiment failed with exit code $EXIT_CODE. Retrying..."
    RETRY_COUNT=$((RETRY_COUNT+1))
    sleep 2  # Optional: wait a bit before retrying
  fi
done

echo "Experiment failed after $MAX_RETRIES attempts."
exit 1
