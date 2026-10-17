#!/usr/bin/env bash
# Does a Dino policy help on Flappy Bird?
#
# For each amount of Flappy data, train three ways and evaluate on the same 50 seeds:
#   scratch   random init
#   dino_ft   init from the Dino policy, fine-tune everything
#   dino_head init from the Dino policy, freeze the conv layers, train only the head
# Plus the Dino policy zero-shot.
#
# Usage: experiments/flappy_transfer.sh [data_dir]   (run from the repo root)
# Runs that already have results are skipped, so the script can be resumed or
# extended, e.g. SEEDS="2 3 4" METHODS="scratch dino_ft" experiments/flappy_transfer.sh
set -euo pipefail

DATA=${1:-data/flappy}
DINO=checkpoints/dino.pt
OUT=runs/transfer
SIZES=${SIZES:-"3600 9000 18000 0"}   # frames: 2, 5, 10 min at 30 Hz; 0 = all
SEEDS=${SEEDS:-"0 1"}
METHODS=${METHODS:-"scratch dino_ft dino_head"}
GAMES=${GAMES:-50}
MAX_STEPS=${MAX_STEPS:-9000}            # 5 min of game time

mkdir -p "$OUT/models"

if [[ ! -f "$OUT/zero_shot/results.json" ]]; then
  python -m pixels_to_keys.evaluate --game flappy --model "$DINO" --games "$GAMES" \
      --max-steps "$MAX_STEPS" --out "$OUT/zero_shot" | tail -n 1
fi

for size in $SIZES; do
  for seed in $SEEDS; do
    for method in $METHODS; do
      name="${method}_f${size}_s${seed}"
      [[ -f "$OUT/$name/results.json" ]] && continue
      extra=()
      [[ $method != scratch ]] && extra+=(--init "$DINO")
      [[ $method == dino_head ]] && extra+=(--freeze-conv)
      [[ $size != 0 ]] && extra+=(--max-frames "$size")
      echo "== $name"
      python -m pixels_to_keys.train --data "$DATA" --out "$OUT/models/$name.pt" \
          --seed "$seed" "${extra[@]}" | tail -n 2
      python -m pixels_to_keys.evaluate --game flappy --model "$OUT/models/$name.pt" \
          --games "$GAMES" --max-steps "$MAX_STEPS" --out "$OUT/$name" | tail -n 1
    done
  done
done

python experiments/plot_transfer.py --runs "$OUT" --data "$DATA"
