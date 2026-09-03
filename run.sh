#!/bin/bash

datasets=(
  "oxford_flowers"
  "oxford_pets"
  "dtd"
  "caltech101"
  "fgvc_aircraft"
  "eurosat"
  "cifar10_custom"
  "stl10_custom"
)

strategies=(
  "centroid"
  "entropy"
  "confidence"
  "margin"
  "margin_confidence"
)

seeds=(2 42 80)
for strategy in "${strategies[@]}"; do
  for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "=========================================="
      echo "Strategy: $strategy | Dataset: $dataset | Seed: $seed"
      echo "=========================================="
      bash scripts/alvlm/main.sh "$dataset" vit_b16 cbsq "$seed" none "$strategy"
    done
  done
done