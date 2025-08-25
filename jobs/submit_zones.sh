#!/bin/bash
MAX_ZONE=19
BLOCK_SIZE=5

for (( start=0; start<=MAX_ZONE; start+=BLOCK_SIZE )); do
  end=$(( start + BLOCK_SIZE - 1 ))
  if (( end > MAX_ZONE )); then
    end=$MAX_ZONE
  fi

  echo "sending ${start}-${end}..."
  sbatch --array=${start}-${end}%${BLOCK_SIZE} run_jobs.sbatch
done
