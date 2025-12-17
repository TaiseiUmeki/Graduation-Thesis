# TrueCoding System

粘土の中間生成物から最終的な創作物へ変換するインタラクティブ画像生成システムのバックエンド実装。

## 概要

このシステムは、ユーザーが粘土で作った中間生成物の画像とクエリを入力し、対話的に解釈を洗練させながら、最終的な創作物の画像を生成します。

## システムフロー

```
A: 入力
  ↓
B: クエリ解釈（複数解釈案の生成・選択）
  ↓
C: 特徴ベクトル生成（属性空間への変換）
  ↓
D: 画像生成と批評（制約追加による反復改善）
  ↓
終了
```

## ディレクトリ構成

```
truecoding/
├── main.py                     # メインエントリーポイント
├── config.py                   # 設定管理
├── attribution.py              # 既存の属性定義
├── models/                     # データモデル
│   ├── __init__.py
│   ├── attribute_space.py      # 属性空間定義
│   ├── session.py              # セッション管理
│   └── constraints.py          # 制約管理
├── engines/                    # 処理エンジン
│   ├── __init__.py
│   ├── query_interpreter.py    # B: クエリ解釈エンジン
│   ├── vector_generator.py     # C: 特徴ベクトル生成
│   └── image_generator.py      # D: 画像生成エンジン
├── utils/                      # ユーティリティ
│   ├── __init__.py
│   ├── openai_client.py        # OpenAI APIラッパー
│   └── image_utils.py          # 画像処理
└── output/                     # 出力ディレクトリ（自動生成）
    ├── images/                 # 生成画像
    └── sessions/               # セッション保存
```

## セットアップ

### 1. 必要なパッケージのインストール

```bash
pip install openai pillow requests
```

### 2. OpenAI API キーの設定

環境変数に設定：

```bash
export OPENAI_API_KEY='your-api-key-here'
```

または、コード内で直接設定：

```python
from main import TrueCodingSystem
system = TrueCodingSystem(api_key='your-api-key-here')
```

## 使用方法

### 基本的な使い方

```python
from main import TrueCodingSystem

# システムを初期化
system = TrueCodingSystem()

# フェーズA: セッション開始
session = system.start_session(
    image_path="path/to/clay_image.jpg",
    query="木製で温かみのある感じにしたい"
)

# フェーズB: クエリ解釈
interpretations = system.interpret_query()
# 解釈案が表示されるので、1つ選択
system.select_interpretation(interpretation_id=2)

# フェーズC: 特徴ベクトル生成
vector_info = system.generate_vector()

# フェーズD: 画像生成
image_path = system.generate_image()
print(f"生成画像: {image_path}")

# 画像を分析
analysis = system.analyze_current_image()

# 制約を追加して再生成
system.add_constraint(
    attribute_key="material:wood_oak",
    constraint_type="greater_than",
    value=0.8,
    description="オーク材の特徴を強く出す"
)
new_image_path = system.generate_image()
```

### クエリの詳細化

解釈案が満足できない場合、クエリを詳細化：

```python
# 追加の詳細を提供
system.refine_query("北欧風のデザインで、丸みを帯びた形状")

# 再度解釈
interpretations = system.interpret_query()
```

### 画像付きでの詳細化

文章だけでなく、画像でも補足可能：

```python
# 補足用の画像を追加
system.interpret_query(
    additional_image="path/to/reference_image.jpg",
    use_image_context=True
)
```

## 属性空間

システムは以下の属性グループを持つ多次元空間で特徴を管理：

- **material**: 材質（石、木、金属、プラスチックなど）
- **finish**: 仕上げ（磨き、マット、艶ありなど）
- **shape**: 形状（球、円柱、流線型など）
- **size**: サイズ・スケール
- **color**: 色・配色
- **texture**: 質感・触感
- **structure**: 構造・構成
- **function**: 機能・アフォーダンス
- **style**: スタイル・時代観
- **visual**: 視覚的演出
- **pattern**: 模様・モチーフ

各属性は 0.0〜1.0 の重みを持ちます。

## 制約の種類

- `less_than`: 属性値 ≤ 指定値
- `greater_than`: 属性値 ≥ 指定値
- `equal`: 属性値 = 指定値
- `range`: 最小値 ≤ 属性値 ≤ 最大値

## セッション管理

セッションは自動的に`output/sessions/`に保存されます。

```python
# セッション情報の取得
summary = system.get_session_summary()
print(summary)

# セッションの読み込み（別の機会に継続する場合）
from models.session import Session
session = Session.load(Path("output/sessions/session_id.json"))
```

## API 使用量の注意

このシステムは OpenAI API を多用します：

- GPT-4: クエリ解釈、特徴ベクトル生成、画像分析
- GPT-4 Vision: 画像理解
- DALL-E 3: 画像生成

コストに注意して使用してください。

## 参考ソースコード

実装の参考にしたソースコード：

- `Grounded-Segment-Anything/chatbot.py` - エージェントシステム設計
- `Grounded-Segment-Anything/automatic_label_demo.py` - 画像理解と GPT 統合
- `open_api/1028_gpt-image-1.py` - GPT-4 Vision 使用例
- `open_api/1021_mask.py` - 画像編集機能

## トラブルシューティング

### API キーエラー

```
ValueError: OpenAI API keyが設定されていません
```

→ 環境変数`OPENAI_API_KEY`を設定してください

### 画像生成エラー

DALL-E 3 の制限：

- サイズ: 1024x1024, 1792x1024, 1024x1792 のみ
- プロンプトは自動的に英語に翻訳・拡張されます

### JSON 解析エラー

GPT のレスポンスが期待と異なる場合、自動的にフォールバック処理が動作します。

## 今後の拡張案

- フロントエンド UI の追加
- 画像編集機能（マスク使用のインペインティング）
- 属性の動的学習
- 複数画像の同時生成と比較
- ユーザーフィードバックの学習

## ライセンス

研究用途
