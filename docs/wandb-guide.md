# WandB 運用ガイド (方式 B)

Claude Code がほぼ全自動で WandB に実験を記録するための設計・導入手順・運用規約。
2026-07-27 の調査 (一次情報を検証済み) に基づく。

## 1. 設計概要

**方式 B**: ユーザー本人の W&B Pro アカウント内に専用チームを作り、そのチームに
スコープされた service account のキーだけをエージェント側に渡す。

```
ユーザーの W&B org (Pro, 100 GB/月)
└── チーム suisho  (= entity。実験データはすべてここに蓄積)
    ├── project suisho-test  (検証用。検証完了後に本番 project へ移行)
    └── service account (team-scoped。シートを消費しない)
          └── API キー → Claude Code の credential masking 経由でのみ使用
```

ユーザー本人 (Admin) のキーはエージェントが到達できる環境に一切置かない。
Admin キーはチーム内の全削除権限を持つため。

### 防御層 (外側から)

1. **アカウントスコープ**: service account はチーム外にアクセス不可。最悪ケースでも
   被害は専用チーム内の実験データに限定される (再生成可能、ローカルにも二重記録あり)
2. **credential masking**: sandbox 内のプロセスはキーの実物を見られない
   (sentinel 値に置換され、wandb 向け通信時のみ proxy が実キーを注入)
3. **sandbox ネットワーク許可リスト**: wandb ドメインのみ追加許可
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

credential masking は**ユーザーレベル設定でのみ有効** (プロジェクト設定の mask 指定は
無視される。Claude Code v2.1.199+、現環境は v2.1.220)。以下をマージする:

```json
{
  "env": {
    "WANDB_API_KEY": "<service account のキー>"
  },
  "sandbox": {
    "network": {
      "tlsTerminate": {},
      "allowedDomains": ["wandb.ai", "*.wandb.ai"]
    },
    "credentials": {
      "envVars": [
        { "name": "WANDB_API_KEY", "mode": "mask", "injectHosts": ["api.wandb.ai"] }
      ]
    }
  }
}
```

- `tlsTerminate: {}` が無いと masking は fail-closed (sentinel がサーバーに届き認証失敗)
- `*.wandb.ai` は apex (`wandb.ai` 本体) を含まないため両方列挙する
- ⚠️ **credential masking は 2026-07-27 時点で発効していない** (v2.1.220)。settings の
  `env` ブロック経由・シェル環境経由の両方で実測したが、sandbox 内に実キーが見えた。
  一方 `network.allowedDomains` は同じ設定ファイルから正しく効いている (許可外ドメインは
  遮断を確認) ため、設定の書き方ではなく Claude Code 側の不具合の可能性が高い。
  `claude --debug` / `/doctor` での調査と upstream への issue 報告を推奨。
  発効しない間、キー漏えい耐性は他の防御層 (SA スコープ + hook + denyRead) に依存する

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
- **sweep 禁止**: `WANDB_MODE` を無視してクラウド同期する既知バグあり
  ([wandb/wandb#6234](https://github.com/wandb/wandb/issues/6234))
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

- [ ] **NG** sandbox 内の `$WANDB_API_KEY` が**実キーのまま** (sentinel になっていない)。
      settings `env` 経由・シェル環境経由の両方で再現。network 分離は有効
      (許可外ドメイン遮断を確認) なので Claude Code 側の不具合の可能性が高い (§2.2)。
      他の防御層 (SA スコープ, hook, env 固定) は有効なため運用は開始可能だが、
      キー漏えい耐性は未達
- [x] `init_run()` でログ → `suisho/suisho-test` に run 出現
      (https://wandb.ai/suisho/suisho-test/runs/a1ge23zd)
- [x] SA キーで他 entity への書き込み → 正しく拒否 (WandbApiFailedError)
- [x] SA キーでの run 削除の実測: **削除できる** (自分で作った捨て run に対し成功)。
      エージェントの全 run は SA 作成のため、実質「チーム内の agent run は全て削除可能」
      が確定した blast radius
- [x] `WANDB_USER_EMAIL` 帰属付き `run.alert()` → 送信は成功。**配信 (受信) はユーザー確認待ち**
- [x] hook ガード: 許可/ブロックを `experiments/004-wandb-verify/hook_tests.sh` で
      再現可能にした (迂回ケースを含む 24 ケース)

## 5. 未確定事項 (2026-07-27 時点)

| 項目 | 状態 |
|---|---|
| Pro プランでの service account 提供 | **確認済 (2026-07-27)**: Pro で team-scoped service account を作成できた |
| service account の run 削除権限 | **実測済 (2026-07-27)**: 自分の run を削除**できる** (§4)。artifact 削除は未実測 |
| service account run からの `wandb.alert()` 配信 | 送信は成功 (`WANDB_USER_EMAIL` 帰属付き)。**受信確認待ち**。不達なら W&B Automations を使う |
| credential masking の非発効 | sandbox 内で実キーが見えている (§4)。`env` ブロック経由・シェル環境経由の両方で再現。`/proc/$$/environ` の実測で **exec 時点から実キー** (sentinel 置換が一度も行われていない) を確認、一方 network 分離 (proxy) は同一設定で有効。rc ファイルによる上書きでもない (Codex 相談で仮説を網羅的に排除)。残る候補は feature gating / build regression / 設定の silent unparse — upstream 報告を推奨。**対処: masking は無いものとして扱い、露出の可能性がある SA キーは随時ローテーションする** |
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
