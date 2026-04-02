#!/bin/bash

# Check if an index argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <IDX>"
  exit 1
fi

IDX=$1
MAX_RETRIES=15
RETRY_COUNT=0
ERROR_LOG="error_log.txt"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  echo "Running load_dishwasher experiment with IDX=$IDX (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)..."
  
  python -m rag_based_prompting.evaluation.run_experiment --exp_start_idx "$IDX" --plan_exp
  
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully!"
    exit 0
  else
    ERROR_MSG="Attempt $((RETRY_COUNT+1)) failed with exit code $EXIT_CODE."
    echo "$ERROR_MSG Retrying..."
    
    # Append error message to external txt file
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $ERROR_MSG" >> "$ERROR_LOG"
    
    RETRY_COUNT=$((RETRY_COUNT+1))
    sleep 2  # Optional: wait a bit before retrying
  fi
done

FINAL_ERROR_MSG="Experiment failed after $MAX_RETRIES attempts."
echo "$FINAL_ERROR_MSG"
echo "$(date '+%Y-%m-%d %H:%M:%S') - $FINAL_ERROR_MSG" >> "$ERROR_LOG"
exit 1
