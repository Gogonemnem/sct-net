#!/bin/bash
#SBATCH --job-name=training_fct
#SBATCH --output=log/%x_%j.out
##SBATCH -C v100-32g
#SBATCH --error=log/%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
##SBATCH --partition=gpu_p2s
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=uio@v100


# Cleans out modules loaded in interactive and inherited by default
# module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
# module load cpuarch/amd

# Loading modules
module load pytorch-gpu/py3/2.2.0

cd /gpfswork/rech/uio/urr97zp/fsod-fct

# Echo of launched commands
set -x

# Your script
base_config_name="pvtv2-pvt4"
few_shot_config_name="pvtv2-attentionrpn-pvt4-multirelation"
dataset="dota"

batch_size=8

# Few-shot Finetuning (Two-branch)
few_shot_iterations="10" # (1 2 3 5 10 3-)

# ------------ Automatic Configuration ------------
# accumulation_steps=$((effective_batch_size / batch_size))
dataset_config="configs/fsod/${dataset}.yaml"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
# ----------------- Training -----------------
# ## Pretraining (Single-Branch) 
srun python3 train_net.py --config-file "$dataset_config" --resume \
    --additional-configs configs/fsod/pretraining.yaml configs/fsod/${base_config_name}.yaml\
        -- SOLVER.IMS_PER_BATCH $batch_size \
        OUTPUT_DIR output/${dataset}/${base_config_name}/pretraining/ \
        # SOLVER.ACCUMULATION_STEPS $accumulation_steps \
        2>&1 | tee "logs/pretraining-${base_config_name}.txt"

### Training (Two-branch) --eval-only 
srun python3 train_net.py --config-file "$dataset_config" --resume \
    --additional-configs configs/fsod/training.yaml configs/fsod/${few_shot_config_name}.yaml\
        -- SOLVER.IMS_PER_BATCH $batch_size \
        OUTPUT_DIR output/${dataset}/${few_shot_config_name}/training/ \
        MODEL.WEIGHTS output/${dataset}/${base_config_name}/pretraining/model_final.pth \
        # SOLVER.ACCUMULATION_STEPS $accumulation_steps \
        2>&1 | tee "logs/training-${config_name}.txt"

### Fine-tuning (Two-branch)
for i in "$few_shot_iterations"; do
    dataset_name="${dataset}_train2017_full_${i}shot"
    
    datasets="('${dataset_name}',)"
    test_shots="('${i}',)"
    support_shot=$((i-1))
    
    # Execute the training command with the adjusted dataset name
    srun python3 train_net.py --config-file "$dataset_config" --resume \
        --additional-configs configs/fsod/finetuning.yaml configs/fsod/${few_shot_config_name}.yaml\
        -- SOLVER.IMS_PER_BATCH $batch_size \
        OUTPUT_DIR output/${dataset}/${few_shot_config_name}/finetuning/${i}shot \
        MODEL.WEIGHTS output/${dataset}/${few_shot_config_name}/training/model_final.pth \
        INPUT.FS.SUPPORT_SHOT $support_shot \
        DATASETS.TRAIN "$datasets" \
        DATASETS.TEST_SHOTS "$test_shots" \
        2>&1 | tee "logs/finetuning-${config_name}_${i}shot.txt"
done