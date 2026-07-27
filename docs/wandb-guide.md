# WandB 運用ガイド

Claude Code がほぼ全自動で WandB に実験を記録するための設計・導入手順・運用規約。
2026-07-27 の調査 (一次情報を検証済み) に基づく。

## 1. 設計概要

ユーザー本人の W&B Pro アカウント内に専用チームを作り、そのチームにスコープされた
service account のキーだけをエージェント側に渡す構成
(導入時の設計検討では「方式 B」と呼んでいた案。呼称は当時のチャット上の議論のみで
使われたもので、本ガイドでは以後使わない)。

```
ユーザーの W&B org (Pro, 100 GB/月)
└── チーム suisho  (= entity。実験データはすべてここに蓄積)
    ├── project suisho-test  (検証用。検証完了後に本番 project へ移行)
    └── service account (team-scoped。シートを消費しない)
          └── API キー → sandbox 有効環境でのみエージェントに供給
```

ユーザー本人 (Admin) のキーはエージェントが到達できる環境に一切置かない。
Admin キーはチーム内の全削除権限を持つため。

### 防御層 (外側から)

1. **アカウントスコープ**: service account はチーム外にアクセス不可。最悪ケースでも
   被害は専用チーム内の実験データに限定される (再生成可能、ローカルにも二重記録あり)
2. **credential masking**: 現在は**不使用** — 機構自体は作動確認済みだが wandb SDK の
   クライアント側キー検証と非互換 (§2.2)。upstream 改善待ち
3. **sandbox ネットワーク許可リスト**: wandb ドメインのみ追加許可。masking 不使用でも
   これによりキーは wandb ドメイン以外へ物理的に送信できない
4. **hook ガード** (`.claude/hooks/wandb-guard.sh`): `wandb login` / `wandb sweep` /
   `wandb artifact delete` / `WANDB_API_KEY` `WANDB_ENTITY` `WANDB_PROJECT`
   `WANDB_BASE_URL` `WANDB_RUN_ID` のインライン設定をブロック
5. **env 固定**: `WANDB_ENTITY` / `WANDB_PROJECT` を settings.json の env で固定
6. **コード規約**: 全スクリプトが `tools/wandb_utils.py` 経由でログ
   (新規 run id 強制、ローカル JSONL 二重記録、checkpoint TTL)

## 2. 導入手順 (ユーザー作業)

### 2.1 W&B 側

1. org を Pro にアップグレード (ユーザーアイコン → Billing)
2. チーム `suisho` を作成 (済 2026-07-27)
3. Team settings → **Service Accounts** → New Team Service Account →
   Authentication Method = Generate API key (済 2026-07-27)
   - ⚠️ Pro での service account 提供は pricing ページ (2025-02 以降一貫して記載) に
     基づく。もし UI に項目が無ければ**フォールバック**: 無料の捨てアカウントを作り、
     チームに Member として追加してそのキーを使う (Member は自分の run しか削除できない)
4. プロジェクト `suisho-test` をチーム内に作成 (可視性: Team):
   wandb.ai 左上の entity 切り替えでチーム `suisho` を選択 → Projects →
   **Create new project** → 名前 `suisho-test`、Privacy = **Team** → Create。
   (直接 URL: `https://wandb.ai/new-project?entity=suisho`)
   検証完了後、本番用 project を同じ手順で作り env の `WANDB_PROJECT` を切り替える

### 2.2 ユーザーレベル `~/.claude/settings.json`

現行の推奨設定 (masking は使わない — 理由は下記):

```json
{
  "env": {
    "WANDB_API_KEY": "<service account のキー>"
  },
  "sandbox": {
    "network": {
      "allowedDomains": ["wandb.ai", "*.wandb.ai"]
    }
  }
}
```

- `*.wandb.ai` は apex (`wandb.ai` 本体) を含まないため両方列挙する
- masking を将来試す場合の参考: `credentials.envVars` の `mode: "mask"` 指定は
  ユーザーレベル設定でのみ有効で、`network.tlsTerminate: {}` が必須 (v2.1.199+)
- **sandbox 導入の経緯 (2026-07-27 解決)**: 障害は 2 段階あった —
  (1) socat 未インストール → Claude Code が sandbox 全体を静かに無効化、
  (2) socat 導入後も Ubuntu の AppArmor 制限
  (`kernel.apparmor_restrict_unprivileged_userns = 1`) で bwrap が起動不能。
  システム管理者による対応で解消し、sentinel 置換・ネットワーク分離・
  filesystem 分離の作動を実測確認済み
- ⚠️ **credential masking は wandb SDK と非互換** (2026-07-27 実測):
  sentinel 置換自体は作動するが、wandb SDK がキー形式を**クライアント側で検証**
  (40 文字以上等) するため、sentinel が API リクエスト前に拒否され、proxy の
  実キー注入まで到達しない。**当面 mask エントリは外して運用**する —
  それでも network allowlist によりキーは wandb ドメイン以外へ送信できず、
  sandbox なし運用より大幅に安全。Claude Code へ format-preserving sentinel の
  feature request を報告予定
- sandbox 有効時の必要設定: `sandbox.filesystem.allowWrite` に `~/.cache/uv/**`
  (uv のロック取得に必要。無いと `uv run` が失敗する)。
  依存欠落の静かな劣化を防ぐため `sandbox.failIfUnavailable: true` も推奨

### 2.3 プロジェクト `.claude/settings.json` への差分

```json
// "env" ブロック (新規追加)
"env": {
  "WANDB_ENTITY": "suisho",
  "WANDB_PROJECT": "suisho-test",
  "WANDB_SILENT": "true"
}

// permissions.allow に追加
"Bash(wandb:*)",

// permissions.deny に追加 (hook と二重の防御)
"Bash(wandb login:*)",
"Bash(wandb sweep:*)",
"Bash(wandb agent:*)",
"Bash(wandb artifact delete:*)",
"Edit(.claude/hooks/wandb-guard.sh)",
"Write(.claude/hooks/wandb-guard.sh)",

// hooks.PreToolUse の Bash matcher の hooks 配列に追加
{
  "type": "command",
  "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/wandb-guard.sh"
}

// sandbox.filesystem.denyWrite に追加
"./.claude/hooks/wandb-guard.sh"
```

### 2.4 `.claude/hooks/command-allowlist.sh` への差分

```bash
# GOVERNED_PREFIXES に追加
"wandb"

# ALLOWED_PATTERNS に追加
# WandB: 同期・状態確認のみ許可 (bare / uv run 経由の両形。
# login/sweep/削除系は wandb-guard.sh でもブロック)
'^(uv run )?wandb sync( --sync-all| --clean| --id [a-zA-Z0-9]+)*( [A-Za-z0-9._/][A-Za-z0-9._/-]*)*$'
'^(uv run )?wandb (status|--version)$'
# 実験スクリプトの実行 (experiments/NNN-name/*.py)
'^uv run python experiments/[0-9]{3}-[a-z0-9-]+/[A-Za-z0-9._-]+\.py( [A-Za-z0-9._/=,-]+)*$'
```

## 3. 運用規約 (エージェント・人間共通)

- ログは必ず `tools/wandb_utils.py` の `init_run()` 経由。`wandb.init()` 直接呼び出し禁止
- **`wandb login` 禁止** (~/.netrc に平文永続化されるため)。キーは masking 経由のみ
- **sweep は当面ブロック** (禁止理由はデータ容量ではない):
  (1) sweeps は `WANDB_MODE` を無視してクラウド同期する既知バグがあり
  ([wandb/wandb#6234](https://github.com/wandb/wandb/issues/6234))、現行の online
  運用では通常 run に実害はないが、**smoke test (mode="disabled") 中でも sweep は
  クラウドに書いてしまう**。(2) sweep が生成する run は `init_run()` の規約
  (新規 run id 強制・ローカル JSONL 二重記録) を通らない。
  将来 sweep を使いたくなったら、専用の規約を設計した上で hook のブロックを解除する
- 動作確認 (smoke test) は `init_run(..., smoke=True)` → mode="disabled" でクラウドへは
  完全 no-op (ローカル JSONL は書かれる。gitignore 済み)
- run id は毎回新規 (`WANDB_RUN_ID` の再利用は resume 上書き事故のもと)
- メトリクス分析はクラウド API ではなく**ローカル JSONL** (`experiments/*/metrics-*.jsonl`)
  を読む (レート制限回避 + 同期遅延の影響なし)
- checkpoint artifact は best / final のみ無期限。中間は TTL 14 日 (100 GB/月 枠の節約。
  artifact はファイル単位で重複排除されるため差分の小さい版は安い)
- 学習発散等の通知は `run.alert()` を呼ぶが、service account run では届かない可能性
  あり (§5)。確実な通知が必要なら W&B Automations (project 単位の Slack/webhook) を設定

## 4. 検証チェックリスト (2026-07-27 実施。スクリプト: `experiments/004-wandb-verify/`)

- [x] sandbox 有効化後の再検証 (2026-07-27): sentinel 置換 **OK** (exec 時点から
      sentinel)、network 分離 **OK** (許可外ドメイン遮断)、filesystem 分離 **OK**
      (~/.cache への書き込み拒否を確認)。ただし masking は wandb SDK の
      クライアント側キー検証と非互換のため mask エントリは外して運用 (§2.2)
- [x] **最終疎通確認 (2026-07-27)**: mask なし + network allowlist の本番構成で、
      sandbox 内から run 作成成功 (suisho/suisho-test/czxp7opf)。`~/.cache/uv` 等の
      allowWrite 追加により `uv run` も sandbox 内で正常動作。**導入完了**
- [x] `init_run()` でログ → `suisho/suisho-test` に run 出現
      (https://wandb.ai/suisho/suisho-test/runs/a1ge23zd)
- [x] SA キーで他 entity への書き込み → 正しく拒否 (WandbApiFailedError)
- [x] SA キーでの run 削除の実測: **削除できる** (自分で作った捨て run に対し成功)。
      エージェントの全 run は SA 作成のため、実質「チーム内の agent run は全て削除可能」
      が確定した blast radius
- [x] `WANDB_USER_EMAIL` 帰属付き `run.alert()` → **受信確認済み (2026-07-27)**。
      service account run からでも alert は届く。必要条件: (1) スクリプト側で
      `WANDB_USER_EMAIL` にチーム member のメールを設定 (tools/wandb_utils.py の
      alert 利用時の規約)、(2) User Settings → Alerts で **wandb.alert()** 行の
      Email トグルを ON (現 UI 名。旧称「Scriptable run alerts」)。
      初回テストの不達は再現せず (遅延または迷惑メール分類の可能性)。
      より高機能な通知が欲しくなったら project Automations (Run status change +
      Slack/webhook) が W&B の推奨
- [x] hook ガード: 許可/ブロックを `experiments/004-wandb-verify/hook_tests.sh` で
      再現可能にした (迂回ケースを含む 23 ケース、レビュー指摘の修正適用後に全 PASS)

## 5. 未確定事項 (2026-07-27 時点)

| 項目 | 状態 |
|---|---|
| Pro プランでの service account 提供 | **確認済 (2026-07-27)**: Pro で team-scoped service account を作成できた |
| service account の run 削除権限 | **実測済 (2026-07-27)**: 自分の run を削除**できる** (§4)。artifact 削除は未実測 |
| service account run からの `wandb.alert()` 配信 | **解決 (2026-07-27)**: `WANDB_USER_EMAIL` 帰属 + wandb.alert() Email トグル ON で受信確認済み (§4) |
| credential masking (**決着 2026-07-27**) | 経緯: socat 欠落→AppArmor 制限→依存解消で sentinel 置換は作動。しかし **wandb SDK のクライアント側キー検証と非互換** (sentinel が 40 文字以上のキー形式チェックで拒否され proxy 注入に到達しない) のため mask は外して運用。sandbox の network/filesystem 分離は有効なので、キーは wandb ドメイン以外へ物理的に送信不能。SA キーの定期ローテーションは継続推奨。upstream へ format-preserving sentinel を要望予定 |
| SaaS のレート制限の具体値 | 非公開 (2023 年の旧値: 無料 50 req/分, 有料 200 req/分)。public API 呼び出しは 1 秒以上間隔を空ける |

## 6. 将来の拡張 (フェーズ 2)

- 公式 W&B MCP サーバーをローカル起動 (`uvx --from git+https://github.com/wandb/wandb-mcp-server
  wandb_mcp_server`) + `WANDB_MCP_READ_ONLY=true` で読み取り専用を保証し、run の
  横断分析をエージェントに開放する (hosted 版はクライアント側から read-only を強制できない)
- W&B Automations で run クラッシュ / メトリクス閾値の Slack 通知

## 7. 参考リンク

- [W&B pricing](https://wandb.ai/site/pricing/) / [Service accounts](https://docs.wandb.ai/platform/hosting/iam/service-accounts) / [Team roles](https://docs.wandb.ai/platform/app/settings-page/teams)
- [Artifact TTL](https://docs.wandb.ai/guides/artifacts/ttl/) / [Alerts](https://docs.wandb.ai/guides/runs/alert/) / [Rate limits](https://docs.wandb.ai/models/track/limits)
- [wandb-mcp-server](https://github.com/wandb/wandb-mcp-server)
- [Claude Code sandboxing (credential masking)](https://code.claude.com/docs/en/sandboxing.md)
