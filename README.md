# SCT-Net: Scale Cross-Transformer Network
SCT-Net is a Few-Shot Object Detection (FSOD) model that leverages multi-scale and cross-scale feature management, enhanced detection heads, selective layer freezing, and the incorporation of Twins Transformers. This repository provides tools to train, evaluate, and reproduce SCT-Net's results on popular datasets like DOTA and DIOR, tailored for challenging object detection in aerial imagery. The code is forked from [FCT](https://github.com/GuangxingHan/FCT).


## Overview

SCT-Net (Selective Cross-Transformer Network) is developed for FSOD scenarios, focusing on challenging tasks like aerial imagery detection. It outperforms several baseline models, including FCT, RetinaNet, and AAF-based frameworks, by applying innovative cross-attention and multi-scale features to efficiently identify objects with limited labeled training data. SCT-Net demonstrates superior performance particularly on datasets containing a variety of object scales, such as DOTA and DIOR.



## Installation

To set up SCT-Net, first clone the repository and install the dependencies found in ```requirements.txt```.

Ensure that Detectron2 is installed from the GitHub repo as the latest version (as of writing in 2024) will lead to compatibility issues.

## Data Preparation

- Mainly the model evaluate on two FSOD benchmarks PASCAL VOC and MSCOCO following the previous work [TFA](https://github.com/ucbdrive/few-shot-object-detection).

- Please prepare the datasets and also the few-shot datasets following [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md)
- Please make sure that the datasets (besides PascalVOC) are in coco format
<!-- - First run the scripts ./prepare_coco_few_shot.py and ./prepare_coco_few_shot_test.py  for creating support image list. Customize the paths in these files according to your needs. The seed 1729 is mainly used for fair comparison with [XQSA](https://github.com/pierlj/aaf_framework). -->
- Please run the scripts in ./datasets
- For a fair comparison, two sets of supports are created separately for training and testing. The testing support set is created specifically from the test set of the dataset.

## Converting pre-trained models into detection formats
The script is can be found in the python notebook ```load-and-save-timm-models.ipynb``` from the timm library.

## Model training and evaluation

- We have three steps for model training, first pre-training the single-branch based model over base classes, then training the two-branch based model over base classes, and finally fine-tuning the two-branch based model over novel classes.
- The full training script can be accessed as:
```
sh scripts/train.sh
```


## Results

- SCT-Net achieves competitive results on challenging datasets such as DOTA and DIOR, particularly with novel classes. Key results include:

- Multi-Scale Feature Handling: Demonstrates significant improvements, achieving higher Average Precision (AP) scores compared to single and cross-scale configurations.

- Cross-Attention Mechanism: Outperforms self-attention in multiple configurations, especially when combined with a multi-scale approach.

- C4 vs C5 Detection Heads: Evaluations revealed that the C4 configuration consistently outperformed the C5 setup, indicating that C4 is the optimal backbone for SCT-Net.


## Contributing
We welcome contributions! Please open an issue or submit a pull request for any improvements or fixes.

