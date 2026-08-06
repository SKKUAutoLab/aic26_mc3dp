#!/bin/bash

# stop at the first error
set -e

# Full path of the current script
THIS=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null||echo $0)
# The directory where current script resides
DIR_CURRENT=$(dirname "${THIS}")                    # .
export DIR_TSS=$DIR_CURRENT                         # .

START_TIME="$(date -u +%s.%N)"
###########################################################################################################
echo "###########################"
echo "STARTING"
echo "###########################"

FOLDER_INPUT=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/warehouse_023.yaml'))['data']['root'])")
FOLDER_OUTPUT=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/warehouse_023.yaml'))['data_writer']['root'])")
POSE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/warehouse_023.yaml'))['reidentifier']['pose_estimator']['model'])")

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

python src/core/run_pipeline.py --config configs/warehouse_023.yaml

python -m src.core.filter_object_noise.run_filter \
  --input      $FOLDER_OUTPUT/mots_multi/Warehouse_023.txt \
  --dataset    $FOLDER_INPUT \
  --pose-model $POSE_MODEL \
  --output     $FOLDER_OUTPUT \
  --gpu        0

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

python src/core/run_pipeline.py --config configs/warehouse_024.yaml

python -m src.core.filter_object_noise.run_filter \
  --input      $FOLDER_OUTPUT/mots_multi/Warehouse_024.txt \
  --dataset    $FOLDER_INPUT \
  --pose-model $POSE_MODEL \
  --output     $FOLDER_OUTPUT \
  --gpu        0

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

python src/core/run_pipeline.py --config configs/warehouse_025.yaml

python -m src.core.box_refinement.refine_scene25 \
  --input   $FOLDER_OUTPUT/mots_multi/Warehouse_025.txt \
  --dataset $FOLDER_INPUT \
  --output  $FOLDER_OUTPUT \
  --split   test \
  --gpu     0 \
  --keep-ply

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

python src/core/run_pipeline.py --config configs/warehouse_026.yaml

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

python src/core/run_pipeline.py --config configs/warehouse_027.yaml

python -m src.core.box_refinement.refine_scene27 \
  --input   $FOLDER_OUTPUT/mots_multi/Warehouse_027.txt \
  --dataset $FOLDER_INPUT \
  --output  $FOLDER_OUTPUT \
  --split   test \
  --gpu     0 \
  --keep-ply

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

python tools/build_submission.py  \
	--result   \
		"$FOLDER_OUTPUT/Warehouse_023/Warehouse_023_filtered.txt"  \
		"$FOLDER_OUTPUT/Warehouse_024/Warehouse_024_filtered.txt"  \
		"$FOLDER_OUTPUT/Warehouse_025/Warehouse_025_refined.txt"  \
		"$FOLDER_OUTPUT/mots_multi/Warehouse_026.txt"  \
		"$FOLDER_OUTPUT/Warehouse_027/Warehouse_027_refined.txt"  \
	--output $FOLDER_OUTPUT/mots_multi/track1.txt

echo "###########################"
echo "CURRENT_TIME: $(date +%Y-%m-%d_%H:%M:%S)"
echo "###########################"

echo "###########################"
echo "ENDING"
echo "###########################"
###########################################################################################################
END_TIME="$(date -u +%s.%N)"

ELAPSED="$(bc <<<"$END_TIME-$START_TIME")"
echo "Total of $ELAPSED seconds elapsed."
