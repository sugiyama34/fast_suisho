#!/bin/bash
# MIG3: 本番規模の BP の複製 (bp-rep, ノイズ床)。BP の最良のレシピ (LR ×0.7, lr-min ×3) で教師の順序だけ回転 (016 始まり)。
# LR の schedule を本番の BP と同じにするため E=20 で回す (e16 / e20 を本番の BP の同じ epoch と比べる)
cd /home/sugiyama/fast_suisho
exec bash experiments/011-pcalm/gpu_queue.sh MIG-0d33d072-e48b-5dd1-8ff1-3d8b3ce6056b f-bp-lr0.7-lrmin3-rot
