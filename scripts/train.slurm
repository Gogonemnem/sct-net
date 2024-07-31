#!/bin/bash
#SBATCH --job-name=training
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

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) dataset="$2"; shift ;;
        --base-config) base_config_name="$2"; shift ;;
        --few-shot-config) few_shot_config_name="$2"; shift ;;
        --num-gpus) num_gpus="$2"; shift ;;
        --batch-size) batch_size="$2"; shift ;;
        --few-shot-iterations) few_shot_iterations="$2"; shift ;;
        --stages) stages="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set default values if not provided
dataset=${dataset:-dota}
base_config_name=${base_config_name:-pvt_v2/pvtv2-pvt5}
few_shot_config_name=${few_shot_config_name:-pvt_v2/pvtv2-attentionrpn-pvt5}
num_gpus=${num_gpus:-8}
batch_size=${batch_size:-8}
few_shot_iterations=${few_shot_iterations:-"10"} # "1 2 3 5 10 3-"
stages=${stages:-"single-branch,two-branch,fine-tuning"}

job_name="${dataset}_${base_config_name}_${few_shot_config_name}"

echo "Job name: ${job_name}"

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



# ------------ Automatic Configuration ------------
dataset_config="configs/datasets/${dataset}.yaml"

pretraining_config="configs/stages/pretraining.yaml"
training_config="configs/stages/training.yaml"

if [[ "$base_config_name" == *"retinanet"* && "$base_config_name" != *"5"* ]]; then
    finetuning_config="configs/stages/finetuning-retinanet.yaml"
else
    finetuning_config="configs/stages/finetuning.yaml"
fi

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ----------------- Training -----------------
IFS=',' read -ra STAGES_ARRAY <<< "$stages"
for stage in "${STAGES_ARRAY[@]}"; do
    if [[ "$stage" == "single-branch" ]]; then
        ### Pretraining (Single-Branch)
        srun python3 train_net.py --config-file "$dataset_config" --resume \
            --additional-configs $pretraining_config configs/${base_config_name}.yaml\
                -- SOLVER.IMS_PER_BATCH $batch_size \
                OUTPUT_DIR output/${dataset}/${base_config_name}/pretraining/ \
                # SOLVER.ACCUMULATION_STEPS $accumulation_steps \
                2>&1 | tee "logs/pretraining-${base_config_name}.txt"
    fi

    if [[ "$stage" == "two-branch" ]]; then
        ### Training (Two-branch) --eval-only
        srun python3 train_net.py --config-file "$dataset_config" --resume \
            --additional-configs $training_config configs/${few_shot_config_name}.yaml\
                -- SOLVER.IMS_PER_BATCH $batch_size \
                OUTPUT_DIR output/${dataset}/${few_shot_config_name}/training/ \
                MODEL.WEIGHTS output/${dataset}/${base_config_name}/pretraining/model_final.pth \
                DATASETS.TEST_SHOTS "(10,)" \
                # SOLVER.ACCUMULATION_STEPS $accumulation_steps \
                2>&1 | tee "logs/training-${config_name}.txt"
    fi

    if [[ "$stage" == "fine-tuning" ]]; then
        ### Fine-tuning (Two-branch)
        for i in $few_shot_iterations; do
            dataset_name="${dataset}_train2017_full_${i}shot"

            datasets="('${dataset_name}',)"
            test_shots="('${i}',)"
            support_shot=$((i-1))

            # Execute the training command with the adjusted dataset name
            srun python3 train_net.py --config-file "$dataset_config" --resume \
                --additional-configs $finetuning_config configs/${few_shot_config_name}.yaml\
                -- SOLVER.IMS_PER_BATCH $batch_size \
                OUTPUT_DIR output/${dataset}/${few_shot_config_name}/finetuning/${i}shot \
                MODEL.WEIGHTS output/${dataset}/${few_shot_config_name}/training/model_final.pth \
                INPUT.FS.SUPPORT_SHOT $support_shot \
                DATASETS.TRAIN "$datasets" \
                DATASETS.TEST_SHOTS "$test_shots" \
                2>&1 | tee "logs/finetuning-${config_name}_${i}shot.txt"
        done
    fi
done