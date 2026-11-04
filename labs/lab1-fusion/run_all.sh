#!/bin/sh
# train and evaluate every modality for every seed, then build the report
set -e
for modality in fusion text audio; do
    for seed in 0 1 2; do
        python train.py --config config.yaml --modality $modality --seed $seed
        python eval.py --config config.yaml --modality $modality --seed $seed
    done
done
python report.py --config config.yaml
pause