# 実験 004: WandB 統合の検証

日付: 2026-07-27
スクリプト: `verify.py` (チェックリスト A〜E), `delete_probe.py` (D の単発再実行)
背景: docs/wandb-guide.md の構成 (専用チーム + service account) 導入に伴う動作・権限の実測。

## 結果

| テスト | 結果 | 詳細 |
|---|---|---|
| A. smoke モード (`mode="disabled"`) | ✅ | 完全 no-op。クラウド送信なし |
| B. 初回オンライン run | ✅ | [suisho/suisho-test/a1ge23zd](https://wandb.ai/suisho/suisho-test/runs/a1ge23zd) に 3 step 記録 |
| C. チーム境界 (他 entity への書き込み) | ✅ 拒否 | `WandbApiFailedError` — SA キーはチーム `suisho` 外に書けない |
| D. SA キーによる run 削除 | ⚠️ **削除可** | 自作の捨て run の削除に成功。公式ドキュメント未記載の権限を実測で確定 |
| E. `run.alert()` (email 帰属付き) | ✅ 送信 | 受信確認はユーザー待ち |
| (別途) credential masking | ❌ 非発効 | sandbox 内で実キーが見える。原因調査中 (docs/wandb-guide.md §5) |

## 考察

- **blast radius の確定**: SA は自分の作った run を削除できる (D)。エージェントの
  run はすべて SA 作成のため、最悪ケースは「チーム `suisho` 内の agent run の削除」。
  チーム外 (個人アカウント・他プロジェクト) には物理的に届かない (C) ため、
  設計どおり被害はチーム内に限定される。
- **masking 非発効**は防御層 2 の欠落であり要修正だが、層 1 (SA スコープ) が
  効いているため運用開始は可能と判断。
- verify.py の初回実行では `run.path` (文字列) を `"/".join()` して
  1 文字ずつ分割するバグがあった (修正済み)。D は `delete_probe.py` で再実測。

## 再現方法

```
uv run python experiments/004-wandb-verify/verify.py
```
(D を単発で再実測する場合は delete_probe.py 内の run path を書き換えて実行)
