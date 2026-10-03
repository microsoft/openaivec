# openaivec-skill 1.3.0 — Business bulk-processing assistant

## 大量のデータを、ふつうの言葉と選択肢で処理

ビジネスユーザー向けに、入口を「目的 → 対象 → 少量の結果確認 →
一括実行」に整理しました。SQLやコード、並列数などの調整は
アシスタントが担当し、利用者には業務上の選択と承認だけを案内します。

- **明示的な呼び出しは不要**：大量の文章入りテーブルや文書群を扱う依頼、
  または許可したデータ確認で見つかった適切な処理機会から活用します。
- **表のテキスト分類が得意**：アンケート、口コミ、問い合わせ、商談メモ、
  商品説明などの分類・抽出・要約・翻訳をまとめて実行します。
- **効率的な一括処理**：重複した入力はまとめて処理し、元の行と識別子を
  保持して復元します。承認済みの試行結果も再利用し、長い処理は進捗を報告します。
- **安全な確認手順**：データ送信、費用のある試行と全件処理、新しい保存先を
  確認します。元ファイルの変更、追加ソフトや中間結果の保存は無断で行いません。
- **日本語ガイドを追加**：コードやコマンドを実行しない導入・利用方法を用意しました。

速度や料金はデータとサービスに依存します。一定の高速化率を保証せず、
数値だけの集計・単純な変換・Excelの装飾を不要にAI処理へ切り替えません。

## ビジネスユーザー向けのインストール方法

GitHub Copilot、Claude Code、Codex、Cursorなどのスキル対応アシスタントで、
作業するプロジェクトを開き、次をチャットに貼ってください。
利用者がコマンドを実行する必要はありません。

```text
Microsoft公式の microsoft/openaivec から openaivec-skill を、
このプロジェクトだけで使えるようにインストールしてください。
公式リリースのうち openaivec-skill-vX.Y.Z というタグの最新安定版を選び、
その正確なタグに固定してください。ライブラリ本体のリリースは選ばないでください。
追加するファイルと影響する範囲を説明し、既存のスキルを置き換える場合は
止めて、インストール前に私の承認を求めてください。
承認後は、このアシスタントに対応した方法でインストールを実行し、
公式の入手元、バージョン、このプロジェクトだけの設定であることを確認してください。
新しいチャットが必要なら教えてください。
業務データを開くこと、処理用ソフトの追加、認証設定、AIへの送信はまだ行わず、
私にコマンド実行を求めないでください。
ここで導入できない場合は、管理者向けの依頼文を作ってください。
```

説明を確認して承認し、必要なら新しいチャットを開きます。
導入後は、スキル名を毎回指定せずに、例えば次のように依頼できます。

```text
このExcelのコメントを、全部まとめて話題と感情で分類してください。
元の回答番号と行は残し、元ファイルは変更しないでください。
まず数件の結果を見せて、選択肢で案内してください。
データの送信と、新しい結果の保存は承認を取ってください。
```

会社のAI接続や追加ソフトが未準備の場合は、アシスタントが影響を説明し、
承認された準備だけを進めます。キーやパスワードをチャットに貼らないでください。
対応する導入機能がない環境では、勝手な方法でインストールせず管理者に引き継ぎます。

[日本語の利用ガイド](https://microsoft.github.io/openaivec/agent-skill/getting-started-ja/) ·
[English getting-started guide](https://microsoft.github.io/openaivec/agent-skill/)

## English: install by asking your assistant

Open the intended workspace in an Agent Skills-compatible assistant and paste:

```text
Install Microsoft's official openaivec-skill from microsoft/openaivec for
this workspace only. Select the latest stable Skill release tagged
openaivec-skill-vX.Y.Z, not a Python-library release or untagged branch,
and pin installation to that exact tag.
Explain the files and scope, stop if an existing Skill would be replaced,
and ask before installing. After approval, use this assistant's supported
Skill installer, perform installation yourself, and verify the source,
version, and workspace-only scope. Tell me if I need a new chat.
Do not open business data, install processing software, configure secrets,
or make an AI request yet. Do not ask me to run commands; prepare an
administrator handoff if direct installation is unavailable.
```

After installation, ask normally: "Classify all the comments in this workbook
by topic and sentiment; preserve the IDs, preview a few rows, and ask before
the full run or a new output." You do not need to invoke the skill explicitly.

## Assistant and maintainer details

The release adds guided-conversation and measured-performance references,
an in-memory bulk runner with separate pilot/full-run gates, deterministic
checkpoint mapping, accepted-pilot reuse, ordered row restoration, and
offline regression coverage. Committed chunks can be reused in the same
process; cross-process checkpoints require separate authorization.
Remote retries can still incur charges; the helper does not claim
exactly-once delivery to the provider.

The portable archives include the English/Japanese guides, references,
helper scripts, evaluation cases, release notes, and MIT license.
This is an **Agent Skill release**, not a new Python-library release.
