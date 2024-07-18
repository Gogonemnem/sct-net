#!/bin/sh

# ----------------- FYI -----------------
# Due to my added flag --additional-configs, be careful with spaces.
# In short, don´t add spaces after the last config file.
# 
# Also, the additional keys overwriting the base config file should be separated "--"
# Thus, the train_net also takes care of this additional "oneven" number of opts
# ----------------- FYI -----------------

# ----------------- Configuration -----------------

base_config_name="pvtv2-pvt5"
few_shot_config_name="pvtv2-attentionrpn-pvt5"
dataset="dota"

num_gpus=1
# effective_batch_size=8
batch_size=1

# Few-shot Finetuning (Two-branch)
few_shot_iterations="10" # (1 2 3 5 10 30) -1

# ------------ Automatic Configuration ------------
# accumulation_steps=$((effective_batch_size / batch_size))
dataset_config="configs/fsod/${dataset}.yaml"

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ----------------- Training -----------------
### Pretraining (Single-Branch)
# python3 train_net.py --num-gpus $num_gpus --resume --dist-url auto \
#     --config-file "$dataset_config" \
#     --additional-configs configs/fsod/pretraining.yaml configs/fsod/${base_config_name}.yaml\
#         -- SOLVER.IMS_PER_BATCH $batch_size \
#         OUTPUT_DIR output/${dataset}/${base_config_name}/pretraining/ \
#         # SOLVER.ACCUMULATION_STEPS $accumulation_steps \
#         2>&1 | tee "logs/pretraining-${config_name}.txt"

# ### Training (Two-branch) --eval-only 
# python3 train_net.py --num-gpus $num_gpus --dist-url auto \
#     --config-file "$dataset_config" \
#     --additional-configs configs/fsod/training.yaml configs/fsod/${few_shot_config_name}.yaml\
#         -- SOLVER.IMS_PER_BATCH $batch_size \
#         OUTPUT_DIR output/${dataset}/${few_shot_config_name}/training/ \
#         # MODEL.WEIGHTS output/${dataset}/${base_config_name}/pretraining/model_final.pth \
#         # SOLVER.ACCUMULATION_STEPS $accumulation_steps \
#         2>&1 | tee "logs/training-${config_name}.txt"

### Fine-tuning (Two-branch)
for i in "$few_shot_iterations"; do
    dataset_name="${dataset}_train2017_full_${i}shot"
    
    datasets="('${dataset_name}',)"
    test_shots="('${i}',)"
    support_shot=$((i-1))
    
    # Execute the training command with the adjusted dataset name
    python3 train_net.py --num-gpus $num_gpus  --dist-url auto \
        --additional-configs configs/fsod/finetuning.yaml configs/fsod/${few_shot_config_name}.yaml\
        --config-file "$dataset_config" \
        -- SOLVER.IMS_PER_BATCH $batch_size \
        OUTPUT_DIR output/${dataset}/${few_shot_config_name}/finetuning/${i}shot \
        MODEL.WEIGHTS output/${dataset}/${few_shot_config_name}/training/model_final.pth \
        INPUT.FS.SUPPORT_SHOT $support_shot \
        DATASETS.TRAIN "$datasets" \
        DATASETS.TEST_SHOTS "$test_shots" \
        2>&1 | tee "logs/finetuning-${config_name}_${i}shot.txt"
done