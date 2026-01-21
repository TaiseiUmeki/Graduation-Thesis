"""
使用例: TrueCoding System のデモスクリプト

このスクリプトは、システムの基本的な使い方を示します。
実際に実行するには、画像ファイルとOpenAI APIキーが必要です。
"""

from main import TrueCodingSystem
from pathlib import Path


def example_basic_flow():
    """基本的なフロー（A→B→C→D）の例"""
    print("=" * 70)
    print("例1: 基本的なフロー")
    print("=" * 70)
    
    # システムを初期化
    system = TrueCodingSystem()
    
    # フェーズA: セッション開始
    print("\n### フェーズA: 入力 ###")
    session = system.start_session(
        image_path="data/images/original_tank.jpg",  # 粘土の画像パス
        query="和風なデザインにしたい",
        concept="tank"
    )
    
    # フェーズB: クエリ解釈
    print("\n### フェーズB: クエリ解釈 ###")
    interpretations = system.interpret_query(use_image_context=True)
    
    # ユーザーが解釈案を選択
    print("\n解釈案を選んでください:")
    while True:
        try:
            choice = input("解釈案の番号を入力 (1-3): ").strip()
            choice_id = int(choice)
            if 1 <= choice_id <= 3:
                system.select_interpretation(interpretation_id=choice_id)
                break
            else:
                print("1～3の数字を入力してください")
        except ValueError:
            print("数字で入力してください")
    
    # フェーズC: 特徴ベクトル生成
    print("\n### フェーズC: 特徴ベクトル生成 ###")
    vector_info = system.generate_vector()
    
    # フェーズD: 画像生成
    print("\n### フェーズD: 画像生成 ###")
    image_path = system.generate_image()
    
    print(f"\n✓ 画像が生成されました: {image_path}")
    
    # 画像を分析
    print("\n画像を分析中...")
    analysis = system.analyze_current_image()
    
    return system


def example_with_constraints():
    """制約を追加しながら改善していく例"""
    print("\n" + "=" * 70)
    print("例2: 制約を追加して反復改善")
    print("=" * 70)
    
    # 前の例から続けると仮定
    # system = ... (前のセッションの続き)
    
    # ここでは新しくシステムを作成
    system = TrueCodingSystem()
    
    # フェーズA〜Dを簡易的に実行
    system.start_session(
        image_path="data/images/original_car.jpg",
        query="未来的で流線型のデザインにしたい",
        concept="car"
    )
    system.interpret_query(use_image_context=True)
    system.select_interpretation(interpretation_id=1)
    system.generate_vector()
    system.generate_image()
    
    print("\n### 1回目の制約追加 ###")
    # オーク材の特徴をもっと強く
    system.add_constraint(
        attribute_key="material:wood_oak",
        constraint_type="greater_than",
        value=0.8,
        description="オーク材の特徴（木目など）をより強調"
    )
    
    # 画像を再生成
    image_path_1 = system.generate_image()
    print(f"✓ 画像を再生成: {image_path_1}")
    
    print("\n### 2回目の制約追加 ###")
    # 角を丸くする
    system.add_constraint(
        attribute_key="shape:rounded_large",
        constraint_type="greater_than",
        value=0.7,
        description="角を大きく丸める"
    )
    
    # 画像を再生成
    image_path_2 = system.generate_image()
    print(f"✓ 画像を再生成: {image_path_2}")
    
    # 最終分析
    print("\n### 最終画像の分析 ###")
    analysis = system.analyze_current_image()
    
    return system


def example_query_refinement():
    """クエリを詳細化していく例"""
    print("\n" + "=" * 70)
    print("例3: クエリの詳細化")
    print("=" * 70)
    
    system = TrueCodingSystem()
    
    # フェーズA
    session = system.start_session(
        image_path="path/to/clay_object.jpg",
        query="もっとかっこよくしたい",  # 曖昧なクエリ
        concept="object"
    )
    
    # フェーズB: 最初の解釈
    interpretations = system.interpret_query()
    
    # どの解釈も満足できない場合
    print("\n解釈案に満足できなかったので、クエリを詳細化...")
    
    # 詳細化1
    system.refine_query("金属的な質感で、エッジが立った形状")
    interpretations = system.interpret_query()
    
    # まだ満足できない場合
    print("\nさらに詳細化...")
    
    # 画像を追加して詳細化
    interpretations = system.interpret_query(
        additional_image="path/to/reference_design.jpg"
    )
    
    # 今度は満足できる解釈が得られたと仮定
    system.select_interpretation(interpretation_id=1)
    
    # 以降は通常のフロー
    system.generate_vector()
    image_path = system.generate_image()
    
    return system


def example_attribute_exploration():
    print("\n" + "=" * 70)
    print("例4: 属性の探索")
    print("=" * 70)
    
    system = TrueCodingSystem()
    
    # フェーズA〜Cを実行
    system.start_session(
        image_path="path/to/clay_object.jpg",
        query="属性を探索したい",
        concept="object"
    )
    system.interpret_query()
    system.select_interpretation(interpretation_id=1)
    system.generate_vector()
    
    # 推薦属性を確認
    if system.session and system.session.recommended_attributes:
        print("\n推薦された属性:")
        from models.attribute_space import ATTR_SPACE
        
        for attr_key in system.session.recommended_attributes[:5]:
            attr_name = ATTR_SPACE.get_attribute_name(attr_key)
            print(f"  - {attr_name} ({attr_key})")
        
        # 興味深い属性に制約を追加
        print("\n興味深い属性 'finish:hammered'（槌目）に制約を追加...")
        system.add_constraint(
            attribute_key="finish:hammered",
            constraint_type="greater_than",
            value=0.6,
            description="槌目の質感を加える"
        )
    
    # 画像生成
    image_path = system.generate_image()
    
    return system


def example_session_management():
    """セッション管理の例"""
    print("\n" + "=" * 70)
    print("例5: セッション管理")
    print("=" * 70)
    
    system = TrueCodingSystem()
    
    # セッション開始
    session = system.start_session(
        image_path="path/to/clay_object.jpg",
        query="テスト",
        concept="object"
    )
    
    # セッション情報の確認
    summary = system.get_session_summary()
    print("\nセッション情報:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # セッションIDを保存
    session_id = session.session_id
    print(f"\nセッションID: {session_id}")
    print(f"セッションファイル: output/sessions/{session_id}.json")
    
    # 後でセッションを読み込む場合
    from models.session import Session
    from config import Config
    
    session_path = Config.SESSIONS_DIR / f"{session_id}.json"
    if session_path.exists():
        loaded_session = Session.load(session_path)
        print(f"\nセッションを読み込みました: {loaded_session.session_id}")
        print(f"現在のフェーズ: {loaded_session.current_phase}")


def example_complete_workflow():
    """完全なワークフローの例（実際に実行可能）"""
    print("\n" + "=" * 70)
    print("完全なワークフロー例（コメントアウトを外して実行）")
    print("=" * 70)
    
    # 以下のコメントを外して、実際の画像パスとAPIキーを設定してください
    
    """
    from main import TrueCodingSystem
    
    # APIキーを設定（環境変数にない場合）
    system = TrueCodingSystem(api_key='your-api-key-here')
    
    # 1. セッション開始
    session = system.start_session(
        image_path="path/to/your/clay_image.jpg",
        query="木製で柔らかい印象の花瓶にしたい",
        concept="vase"
    )
    
    # 2. クエリ解釈
    interpretations = system.interpret_query(use_image_context=True)
    
    # 解釈案を確認して選択（例: 2番目）
    system.select_interpretation(interpretation_id=2)
    
    # 3. 特徴ベクトル生成
    vector_info = system.generate_vector()
    
    # 4. 画像生成
    image_path = system.generate_image()
    print(f"生成画像: {image_path}")
    
    # 5. 分析
    analysis = system.analyze_current_image()
    
    # 6. 制約追加（必要に応じて）
    system.add_constraint(
        attribute_key="material:wood_oak",
        constraint_type="greater_than",
        value=0.7,
        description="オーク材の質感を強調"
    )
    
    # 7. 再生成
    new_image_path = system.generate_image()
    print(f"改善画像: {new_image_path}")
    
    # 8. セッション情報
    summary = system.get_session_summary()
    print(summary)
    """
    
    print("\n上記のコメントを外して実行してください")


if __name__ == "__main__":
    print("TrueCoding System - 使用例")
    print("=" * 70)
    print("\n以下の例を確認してください：")
    print("1. example_basic_flow() - 基本的なフロー")
    print("2. example_with_constraints() - 制約を追加して改善")
    print("3. example_query_refinement() - クエリの詳細化")
    print("4. example_attribute_exploration() - 属性の探索")
    print("5. example_session_management() - セッション管理")
    print("6. example_complete_workflow() - 完全なワークフロー")
    print("\n各関数を個別に実行するか、完全なワークフローを試してください。")
    print("\n注意: 実際に実行するには画像ファイルとOpenAI APIキーが必要です。")
