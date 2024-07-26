#!/bin/bash
datasets=("dota" "dior")
base_configs=("pvt_v2/pvtv2-pvt5" "pvt_v2/pvtv2-retinanet" "pvt_v2/pvtv2-retinanet-pvt5")
few_shot_configs=("pvt_v2/pvtv2-attentionrpn-pvt5" "pvt_v2/pvtv2-retinanet-attentionrpn-multirelation" "pvt_v2/pvtv2-retinanet-attentionrpn-pvt5")
stages="single-branch,two-branch,fine-tuning"

for dataset in "${datasets[@]}"; do
    for i in "${!base_configs[@]}"; do
        base_config="${base_configs[i]}"
        few_shot_config="${few_shot_configs[i]}"
        sbatch scripts/train.slurm --dataset "$dataset" --base-config "$base_config" --few-shot-config "$few_shot_config" --stages "$stages"
    done
done

# XQSA, only if pretraining is done
# few_shot_configs=("pvt_v2/pvtv2-xqsa-pvt5" "pvt_v2/pvtv2-retinanet-xqsa-multirelation" "pvt_v2/pvtv2-retinanet-xqsa-pvt5")
# stages="two-branch,fine-tuning"

# for dataset in "${datasets[@]}"; do
#     for i in "${!base_configs[@]}"; do
#         base_config="${base_configs[i]}"
#         few_shot_config="${few_shot_configs[i]}"
#         sbatch scripts/train.slurm --dataset "$dataset" --base-config "$base_config" --few-shot-config "$few_shot_config" --stages "$stages"
#     done
# done
