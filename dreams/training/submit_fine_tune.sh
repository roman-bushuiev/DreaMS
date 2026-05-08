#!/bin/bash

# Parse arguments
DEBUG=false
if [ "$1" = "debug" ] || [ "$1" = "--debug" ]; then
    DEBUG=true
fi

# Set parameters based on mode
if [ "$DEBUG" = true ]; then
    gpus=1
    time="00:30:00"
else
    gpus=8
    time="24:00:00"
fi

# LUMI
project_id="project_465002061"
work_dir="/scratch/${project_id}/rbushuie/DreaMS-Mol_dev/DreaMS"
partition="standard-g"

outdir="${work_dir}/job_submissions"
mkdir -p "${outdir}"
outfile="${outdir}/submissions.csv"

# Generate random key for a job
job_key=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10 ; echo '')
job_dir="${outdir}/${job_key}"
mkdir -p "${job_dir}"

# Build export string
export_vars="job_dir=${job_dir}"
if [ "$DEBUG" = true ]; then
    export_vars="${export_vars},DEBUG=true"
fi

# Submit
job_id=$(sbatch \
  --account=${project_id} \
  --partition=${partition} \
  --gpus=${gpus} \
  --gpus-per-node="${gpus}" \
  --ntasks-per-node="${gpus}" \
  --nodes=1 \
  --time=${time} \
  --output="${job_dir}/stdout.txt" \
  --error="${job_dir}/errout.txt" \
  --job-name="${job_key}" \
  --export="${export_vars}" \
  "${work_dir}/dreams/training/fine_tune.sh"
)

# Log
submission="$(date),${job_id},${job_key},debug=${DEBUG}"
echo "${submission}" >> "${outfile}"
echo "${submission}"
