# 開始局面集

## start_sfens_ply24.txt

やねうら王互角局面集2025 (24手目局面集)。30,053局面、1行1 SFEN。

- 出典: https://github.com/yaneurao/YaneuraOu/releases/tag/BalancedPositions2025
- 解説: https://yaneuraou.yaneu.com/2025/07/29/yaneuraou-balanced-position-collection-2025/
- License: MIT (配布元による)
- 生成方法 (配布元記載): 水匠10で1局面2億ノード探索し、評価値の絶対値が50以下の
  24手目局面を抽出したもの
- sha256: `a11e3ae7efd34f4c7ad8a7e42c0f610239927970259a57915f06a644cee8d90d`
  (LF 正規化後。配布オリジナルは CRLF で sha256
  `f93098ab6c0cf5eab5f55aaffd48da4421934e08d4c2fcfba68f7b60853ac0a2`、取得日 2026-07-03。
  CRLF のままだと USI コマンドに `\r` が混入するハーネスがあるため LF に正規化して管理する)
- 行は sfen 文字列で辞書順ソートされており、隣接行は 1 手違いの近似重複が多い。
  **先頭から N 行を取る抽出は禁止** — 等間隔ストライドで抽出すること (docs/PLAN.md 参照)
