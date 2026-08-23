#!/bin/zsh
# Parallel lane for the v2 matrix: runs the configs given as "emb:arch" args.
# Same keyed results file as the main chain; disjoint configs, no collisions.
cd "$(dirname "$0")"
export L1_DATA=l1_data_v2 L1_RES=results_v2 OMP_NUM_THREADS=2
for cfg in "$@"; do
  emb="${cfg%%:*}"; arch="${cfg##*:}"
  echo "=== lane: $emb / $arch ===" >> "l1_lane_${LANE_NAME}.log"
  python3 l1_train.py --emb "$emb" --arch "$arch" >> "l1_lane_${LANE_NAME}.log" 2>&1
done
echo "lane ${LANE_NAME} complete" >> "l1_lane_${LANE_NAME}.log"
