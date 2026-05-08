#!/bin/bash
#SBATCH --job-name DreaMS_fine-tuning
#SBATCH --account OPEN-29-57
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 10:00:00

# Activate conda environment
# eval "$(conda shell.bash hook)"
# conda activate dreams

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_465002061/rbushuie/DreaMS-Mol_dev/DreaMS-Mol/.venv-genmol/bin/activate

# Export project definitions
$(python -c "from dreams.definitions import export; export()")

# Move to running dir
cd "${DREAMS_DIR}/training" || exit 3

# Parse debug flag from command line ($1) or environment variable (from sbatch --export)
if [ "$1" = "--debug" ] || [ "${DEBUG}" = "true" ]; then
    DEBUG=true
else
    DEBUG=false
fi

# Hyperparameters — override via env vars (set by submit_fine_tune.sh --export)
LR=${LR:-5e-6}
BATCH_SIZE=${BATCH_SIZE:-4}         # per-GPU batch size (effective = BATCH_SIZE × num_gpus)
TRIPLET_MARGIN=${TRIPLET_MARGIN:-0.1}
RUN_NAME=${RUN_NAME:-MassSpecGym}

# Contrastive fine-tuning
TRAIN_ARGS=(
    --project_name CONTRASTIVE_FINE_TUNING
    --job_key "${RUN_NAME}"
    --run_name "${RUN_NAME}"
    --train_objective contrastive_spec_embs
    --train_regime fine-tuning
    --dformat A
    --model DreaMS
    --lr "${LR}"
    --batch_size "${BATCH_SIZE}"
    --prec_intens 1.1
    --max_epochs 301
    --log_every_n_steps 5
    --seed 3407
    --train_precision 32
    --val_check_interval 0.1
    --save_top_k -1
    --head_depth 0
    --unfreeze_backbone_at_epoch 0
    --dataset_pth "/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data/MassSpecGym_RDKit_SMILES_neighbours_0.05Da.pkl"
    --df_smiles_similarities "/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data/MassSpecGym_RDKit_SMILES_0.05Da_val_smiles_similarities_asymmetric.pkl"
    --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt"
    --n_pos_samples 1
    --n_neg_samples 1
    --triplet_loss_margin "${TRIPLET_MARGIN}"
    --max_peaks_n 100
)

export SLURM_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

if [ "$DEBUG" = true ]; then
    python3 -u train.py "${TRAIN_ARGS[@]}" --num_devices 1
else
    srun --export=ALL --preserve-env python3 -u train.py "${TRAIN_ARGS[@]}" --num_devices 8
fi
