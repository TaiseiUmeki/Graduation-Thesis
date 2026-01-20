# SupportDesigning System

粘土の中間生成物から最終的な創作物へ変換するインタラクティブ画像生成システムです。ユーザーが粘土で作った中間生成物の画像とテキストクエリを入力し、対話的に解釈を洗練させながら、属性空間上で最適な特徴ベクトルを探索し、最終的な創作物の画像を生成します。

## 主な特徴

- **対話的な解釈選択**: クエリから複数の解釈案を生成し、ユーザーが選択可能
- **属性空間への変換**: 画像を高次元の属性ベクトルで表現
- **反復的な制約追加**: 特徴を段階的に調整して画像を改善
- **探索木による履歴管理**: ノード形式で改善の過程を記録
- **斥力機能**: クローズドノードから離れた新規ベクトルを自動探索

## システムフロー（A→B→C→D）

```
┌─────────────────────────────────────────────────────────┐
│ フェーズA: 入力                                          │
│ - 粘土画像とクエリを入力                               │
│ - セッション開始                                       │
└─────────────────────┬─────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ フェーズB: クエリ解釈                                    │
│ - GPT-4がクエリから複数の解釈案を生成                  │
│ - ユーザーが最適な解釈を選択                           │
│ - 必要に応じてクエリを詳細化して再解釈                │
└─────────────────────┬─────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ フェーズC: 特徴ベクトル生成                              │
│ - 解釈案を属性空間上のベクトルに変換                   │
│ - 主要属性と関連属性を推薦                             │
│ - 探索木のルートノードを作成                           │
└─────────────────────┬─────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│ フェーズD: 画像生成と反復改善                            │
│ - 特徴ベクトルから画像を生成（DALL-E 3）              │
│ - 属性に制約を追加                                     │
│ - ベクトルを更新（クローズドノードから斥力を適用）    │
│ - 新規ベクトルから画像を再生成                         │
│ - 1-4を繰り返す                                        │
└─────────────────────────────────────────────────────────┘
```

## ディレクトリ構成

```
truecoding/
├── main.py                     # メインエントリーポイント（TrueCodingSystem）
├── config.py                   # 設定管理
├── examples.py                 # 使用例
├── requirements.txt            # 依存パッケージ
├── models/                     # データモデル
│   ├── attribute_space.py      # 属性空間定義（7グループ, 41属性）
│   ├── session.py              # セッション管理（探索木含む）
│   └── constraints.py          # 制約管理
├── engines/                    # 処理エンジン
│   ├── query_interpreter.py    # フェーズB: クエリ解釈
│   ├── vector_generator.py     # フェーズC: ベクトル生成と斥力
│   └── image_generator.py      # フェーズD: 画像生成
├── utils/                      # ユーティリティ
│   ├── openai_client.py        # OpenAI APIラッパー
│   └── image_utils.py          # 画像処理
├── data/                       # データディレクトリ
│   ├── images/                 # 入力画像
│   ├── outputs/                # 出力画像
│   └── sessions/               # セッションファイル
└── tests/                      # テストディレクトリ
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主な依存パッケージ：

- openai: ChatGPT, GPT-4 Vision, DALL-E 3
- pillow: 画像処理
- numpy: 数値計算
- requests: HTTP 通信

### 2. OpenAI API キー設定

環境変数で設定（推奨）：

```bash
export OPENAI_API_KEY='sk-...'
```

または `.env` ファイルに記述：

```
OPENAI_API_KEY=sk-...
```

## 使用方法

### ターミナルでのフロー実行方法

```
python -i main.py
```

これを実行すると一連のフローが実行され、1 回目の画像生成後対話モードに移行。制約が追加できる。
画像とクエリは main.py の 519 行目に入力できる。

```
1. system.generate_image() で画像生成
2. system.add_constraint(...) で制約追加と再生成
```

制約追加のフォーマット

```
>>> system.add_constraint(
...     attribute_key="shape:pointed",
...     constraint_type="greater_than",
...     value=0.85,description="尖っている形状をより強調")
```

### 最も簡単な実行例

```python
from main import TrueCodingSystem
from pathlib import Path

# システム初期化
system = TrueCodingSystem()

# フェーズA: セッション開始
system.start_session(
    image_path="data/images/original_tank.jpg",
    query="木製で温かみのある雰囲気にしたい"
)

# フェーズB: クエリ解釈
interpretations = system.interpret_query(use_image_context=True)
print("\n解釈案:")
for interp in interpretations:
    print(f"  [{interp.id}] {interp.text}")

# ユーザーが解釈案を選択
system.select_interpretation(interpretation_id=1)

# フェーズC: 特徴ベクトル生成
vector_info = system.generate_vector()

# フェーズD: 画像生成
image_path = system.generate_image()
print(f"\n生成画像: {image_path}")
```

### 反復的な改善（制約追加）

```python
# 制約を追加してベクトルを更新・再生成
system.add_constraint(
    attribute_key="material:wood_oak",
    constraint_type="greater_than",
    value=0.8,
    description="オーク材の特性を強調"
)

# 新規ベクトルで画像再生成
new_image_path = system.generate_image()

# さらに別の属性に制約
system.add_constraint(
    attribute_key="finish:polished",
    constraint_type="greater_than",
    value=0.6,
    description="磨き上げられた質感"
)

newer_image_path = system.generate_image()
```

### クエリの詳細化

初期解釈が満足できない場合：

```python
# クエリを詳細化
system.refine_query("北欧風のミニマルなデザイン、円形の形状で")

# 再度解釈
new_interpretations = system.interpret_query()
system.select_interpretation(interpretation_id=2)

# ベクトル再生成と画像生成
system.generate_vector()
system.generate_image()
```

### 画像付きでの詳細化

参考画像を提供して解釈を改善：

```python
# 追加の参考画像を提供
system.interpret_query(
    additional_image="data/images/reference.jpg",
    use_image_context=True
)

# 以降の処理は同じ
```

## 探索木（Exploration Tree）機能

制約追加により特徴ベクトルを更新する過程を「探索木」として管理します。

### ノード構成

各ノードは以下を保持：

- **node_id**: ノード識別子
- **parent_id**: 親ノード ID（ルートは None）
- **vector**: そのノードの特徴ベクトル
- **constraints**: 適用されている制約のリスト
- **generated_image_path**: 生成された画像
- **is_closed**: 探索終了フラグ
- **timestamp**: 作成時刻
- **note**: メモ（例: "root", "add_constraint:material:wood_oak"）

### ノード操作

```python
# ノード一覧表示
nodes = system.list_nodes()
for node in nodes:
    print(f"Node {node['node_id']}: parent={node['parent_id']}, closed={node['is_closed']}")

# 過去のノードに戻る
system.revert_to_node(node_id=2)

# ノードをクローズド（探索終了）にマーク
system.mark_node_closed(node_id=3)

# クローズドノードに制約を追加すると、
# 斥力が自動的に新規ベクトルに適用され、
# クローズドノードから離れた領域を探索する
system.add_constraint(
    attribute_key="color:warm_red",
    constraint_type="greater_than",
    value=0.7
)
```

## 斥力機能（Repulsion）

クローズドノード（探索終了したノード）から遠い位置で新しいベクトルを探索します。

### 動作原理

```
新規ベクトル V と クローズドベクトル C の間の
ユークリッド距離の2乗 d² が設定値より小さい場合、
差分ベクトル（V - C）の方向に調整力を加えます。

d² < min_squared_distance の場合：
  新規値 = clip(現在値 + repulsion_strength × (V - C), 0.0, 1.0)
```

### パラメータ

`main.py` の `add_constraint` メソッド内（約 375-378 行目）：

```python
closed_vectors = [node.vector for node in closed_nodes]
updated_vector = self.vector_generator.apply_repulsion(
    updated_vector,
    closed_vectors,
    min_squared_distance=0.5,   # 距離の2乗の閾値（調整可能）
    repulsion_strength=0.3      # 斥力の強さ 0.0-1.0（調整可能）
)
```

デフォルト値を調整することで、斥力の強さを制御できます：

- `min_squared_distance` を小さくする → より厳密に離れる
- `repulsion_strength` を大きくする → より強い斥力

## 属性空間

システムは以下の 7 グループ、41 の物理的形状属性を持つ多次元空間を利用：

| グループ    | 説明           | 例                                                                                       |
| ----------- | -------------- | ---------------------------------------------------------------------------------------- |
| **form**    | 基本形態       | 立方体、円柱状、球状、円錐状、板状、中空、中実、網状、骨組み構造、殻構造、単一塊、複合体 |
| **line**    | 輪郭と線       | 直線的、曲線的、尖鋭、丸み、有機的ライン、幾何学的ライン                                 |
| **edge**    | エッジ処理     | 鋭利なエッジ、面取り、丸面取り、ベベル加工                                               |
| **surface** | 表面特性       | 平坦、起伏、凹凸、テクスチャ、光沢、マット、半透明、反射                                 |
| **balance** | バランス・配置 | 対称、非対称、上重心、下重心、中央配置、オフセット配置                                   |
| **texture** | 質感・触覚     | なめらか、ざらつき、粗い、柔軟、硬質、弾性                                               |

各属性は -1.0 ～ 1.0 の重みで表現されます（正の値は特徴を強調、負の値は特徴を回避）。

## 制約タイプ

| タイプ         | 意味       | 例                                                         |
| -------------- | ---------- | ---------------------------------------------------------- |
| `greater_than` | 値以上     | `greater_than, value=0.7` → 属性値 ≥ 0.7                   |
| `less_than`    | 値以下     | `less_than, value=0.3` → 属性値 ≤ 0.3                      |
| `equal`        | 値に等しい | `equal, value=0.5` → 属性値 = 0.5                          |
| `range`        | 範囲内     | `range, min_value=0.4, max_value=0.8` → 0.4 ≤ 属性値 ≤ 0.8 |

## セッション管理

セッションは自動的に `data/sessions/` に JSON 形式で保存されます。

```python
# セッション情報取得
summary = system.get_session_summary()
print(summary)
# 出力例:
# {
#     "session_id": "3815e60a-cf0e-4558-b1b8-fd6be8747604",
#     "current_phase": "D",
#     "num_queries": 2,
#     "num_interpretations": 6,
#     "num_generated_images": 3,
#     "num_constraints": 5,
#     ...
# }

# 以前のセッションを再開
from models.session import Session
from pathlib import Path

session = Session.load(
    Path("data/sessions/3815e60a-cf0e-4558-b1b8-fd6be8747604.json")
)
```

## API 使用料について

このシステムは OpenAI API を多く利用します。使用量に注意してください。

**主な API 呼び出し:**

- **GPT-4**: クエリ解釈、特徴ベクトル生成、ベクトル更新、画像分析
- **GPT-4 Vision**: 画像理解・分析
- **DALL-E 3**: 画像生成（1024x1024）

**コスト削減のコツ:**

- 試行前に `examples.py` の小さな例で動作確認
- API キーの使用量を定期的に確認
- クエリは明確かつ簡潔に記述

## トラブルシューティング

### API キーエラー

```
ValueError: OpenAI API keyが設定されていません
```

**解決方法:**

```bash
export OPENAI_API_KEY='sk-...'
# または
echo "OPENAI_API_KEY=sk-..." > .env
```

### 画像生成エラー（DALL-E 3）

DALL-E 3 の制限：

- **サイズ**: 1024x1024, 1792x1024, 1024x1792 のみ
- **言語**: 自動で英語に翻訳・拡張されます
- **プロンプト長**: 上限は約 1000 文字

### JSON パース エラー

GPT のレスポンス形式が期待と異なる場合、ロギングを確認：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 今後の拡張予定

- [ ] フロントエンド UI（Flask/React）
- [ ] リアルタイムプレビュー
- [ ] 複数画像の同時生成・比較
- [ ] 属性の自動学習
- [ ] インペイント機能（DALL-E Inpaint）
- [ ] ユーザーフィードバック機構

## ライセンス

研究用途

## 参考文献・ソースコード

実装の参考にしたプロジェクト：

- Grounded-Segment-Anything
- OpenAI API Documentation
- DALL-E 3 Guide
