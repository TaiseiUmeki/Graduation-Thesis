"""
session.py の単体テスト
セッション管理機能のテスト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.session import Session, Interpretation, GeneratedImage
from models.attribute_space import AttributeVector
from models.constraints import Constraint, ConstraintType
from datetime import datetime
import tempfile
import json


def test_interpretation_creation():
    """解釈案の作成テスト"""
    print("\n【テスト1】解釈案の作成")
    
    interp = Interpretation(
        id=1,
        text="木製で温かみのあるデザイン",
        reasoning="クエリから木材と温もりのキーワードを抽出"
    )
    
    assert interp.id == 1
    assert interp.text == "木製で温かみのあるデザイン"
    assert interp.reasoning == "クエリから木材と温もりのキーワードを抽出"
    assert isinstance(interp.timestamp, datetime)
    print("✓ 解釈案を作成しました")
    
    # 辞書変換
    dict_data = interp.to_dict()
    assert dict_data["id"] == 1
    assert dict_data["text"] == interp.text
    print("✓ 辞書への変換成功")


def test_generated_image_creation():
    """生成画像情報の作成テスト"""
    print("\n【テスト2】生成画像情報の作成")
    
    vector = AttributeVector(weights={"material:wood_oak": 0.8})
    constraint = Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7
    )
    
    image = GeneratedImage(
        image_path="/path/to/image.png",
        prompt="Oak wood with matte finish",
        vector=vector,
        constraints=[constraint]
    )
    
    assert image.image_path == "/path/to/image.png"
    assert image.prompt == "Oak wood with matte finish"
    assert image.vector is not None
    assert len(image.constraints) == 1
    assert isinstance(image.timestamp, datetime)
    print("✓ 生成画像情報を作成しました")
    
    # 辞書変換
    dict_data = image.to_dict()
    assert dict_data["image_path"] == "/path/to/image.png"
    assert "vector" in dict_data
    assert len(dict_data["constraints"]) == 1
    print("✓ 辞書への変換成功")


def test_session_creation():
    """セッションの作成テスト"""
    print("\n【テスト3】セッションの作成")
    
    session = Session()
    
    assert session.session_id is not None
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)
    assert session.current_phase == "A"
    assert len(session.query_history) == 0
    assert len(session.interpretations) == 0
    print("✓ セッションを作成しました")
    print(f"  Session ID: {session.session_id}")
    print(f"  現在のフェーズ: {session.current_phase}")


def test_session_add_query():
    """クエリ追加のテスト"""
    print("\n【テスト4】クエリの追加")
    
    session = Session()
    
    session.add_query("木製にしたい")
    assert len(session.query_history) == 1
    assert session.query_history[0] == "木製にしたい"
    print("✓ クエリを追加: '木製にしたい'")
    
    session.add_query("さらに温かみを出したい")
    assert len(session.query_history) == 2
    print("✓ クエリを追加: 'さらに温かみを出したい'")


def test_session_add_interpretations():
    """解釈案追加のテスト"""
    print("\n【テスト5】解釈案の追加")
    
    session = Session()
    
    interp1 = Interpretation(
        id=1,
        text="解釈案1",
        reasoning="理由1"
    )
    interp2 = Interpretation(
        id=2,
        text="解釈案2",
        reasoning="理由2"
    )
    
    session.add_interpretations([interp1, interp2])
    assert len(session.interpretations) == 2
    print("✓ 2つの解釈案を追加しました")


def test_session_select_interpretation():
    """解釈案選択のテスト"""
    print("\n【テスト6】解釈案の選択")
    
    session = Session()
    
    interp1 = Interpretation(id=1, text="解釈1", reasoning="理由1")
    interp2 = Interpretation(id=2, text="解釈2", reasoning="理由2")
    session.add_interpretations([interp1, interp2])
    
    # 解釈2を選択
    selected = session.select_interpretation(2)
    assert selected is not None
    assert selected.id == 2
    assert session.selected_interpretation == selected
    print("✓ 解釈案2を選択しました")
    
    # 存在しないIDを選択
    not_found = session.select_interpretation(99)
    assert not_found is None
    print("✓ 存在しないIDの場合はNoneを返します")


def test_session_set_vector():
    """特徴ベクトル設定のテスト"""
    print("\n【テスト7】特徴ベクトルの設定")
    
    session = Session()
    
    vector = AttributeVector(weights={
        "material:wood_oak": 0.8,
        "finish:matte": 0.7
    })
    
    session.set_vector(vector)
    assert session.current_vector is not None
    assert session.current_vector.weights["material:wood_oak"] == 0.8
    print("✓ 特徴ベクトルを設定しました")


def test_session_add_constraint():
    """制約追加のテスト"""
    print("\n【テスト8】制約の追加")
    
    session = Session()
    
    constraint1 = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7
    )
    session.add_constraint(constraint1)
    assert len(session.constraints) == 1
    print("✓ 制約を追加しました")
    
    constraint2 = Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5
    )
    session.add_constraint(constraint2)
    assert len(session.constraints) == 2
    print("✓ 2つ目の制約を追加しました")


def test_session_add_generated_image():
    """生成画像追加のテスト"""
    print("\n【テスト9】生成画像の追加")
    
    session = Session()
    
    image1 = GeneratedImage(
        image_path="/path/to/image1.png",
        prompt="Prompt 1"
    )
    session.add_generated_image(image1)
    assert len(session.generated_images) == 1
    print("✓ 生成画像を追加しました")
    
    image2 = GeneratedImage(
        image_path="/path/to/image2.png",
        prompt="Prompt 2"
    )
    session.add_generated_image(image2)
    assert len(session.generated_images) == 2
    print("✓ 2つ目の生成画像を追加しました")


def test_session_get_latest_image():
    """最新画像取得のテスト"""
    print("\n【テスト10】最新画像の取得")
    
    session = Session()
    
    # 画像がない場合
    latest = session.get_latest_image()
    assert latest is None
    print("✓ 画像がない場合はNoneを返します")
    
    # 画像を追加
    image1 = GeneratedImage(image_path="/path/to/image1.png", prompt="P1")
    image2 = GeneratedImage(image_path="/path/to/image2.png", prompt="P2")
    session.add_generated_image(image1)
    session.add_generated_image(image2)
    
    latest = session.get_latest_image()
    assert latest == image2
    assert latest.image_path == "/path/to/image2.png"
    print("✓ 最新の画像を取得しました")


def test_session_get_active_constraints():
    """アクティブな制約取得のテスト"""
    print("\n【テスト11】アクティブな制約の取得")
    
    session = Session()
    
    c1 = Constraint(
        attribute="attr1",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.5,
        is_active=True
    )
    c2 = Constraint(
        attribute="attr2",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.8,
        is_active=False
    )
    
    session.add_constraint(c1)
    session.add_constraint(c2)
    
    active = session.get_active_constraints()
    assert len(active) == 1
    assert active[0].attribute == "attr1"
    print("✓ アクティブな制約のみを取得しました")


def test_session_serialization():
    """セッションのシリアライゼーションテスト"""
    print("\n【テスト12】セッションのシリアライゼーション")
    
    session = Session()
    session.initial_image_path = "/path/to/initial.png"
    session.initial_query = "テストクエリ"
    session.add_query("テストクエリ")
    
    interp = Interpretation(id=1, text="解釈", reasoning="理由")
    session.add_interpretations([interp])
    session.select_interpretation(1)
    
    vector = AttributeVector(weights={"attr1": 0.5})
    session.set_vector(vector)
    
    session.current_phase = "C"
    
    # 辞書に変換
    dict_data = session.to_dict()
    
    assert dict_data["session_id"] == session.session_id
    assert dict_data["initial_image_path"] == "/path/to/initial.png"
    assert dict_data["initial_query"] == "テストクエリ"
    assert len(dict_data["query_history"]) == 1
    assert len(dict_data["interpretations"]) == 1
    assert dict_data["selected_interpretation"] is not None
    assert dict_data["current_vector"] is not None
    assert dict_data["current_phase"] == "C"
    print("✓ セッションを辞書に変換しました")


def test_session_save_and_load():
    """セッションの保存と読み込みテスト"""
    print("\n【テスト13】セッションの保存と読み込み")
    
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # セッションを作成
        original = Session()
        original.initial_image_path = "/path/to/test.png"
        original.initial_query = "保存テスト"
        original.add_query("保存テスト")
        
        interp = Interpretation(id=1, text="解釈", reasoning="理由")
        original.add_interpretations([interp])
        original.select_interpretation(1)
        
        # 保存
        original.save(tmpdir_path)
        save_path = tmpdir_path / f"{original.session_id}.json"
        assert save_path.exists(), "セッションファイルが作成されていません"
        print("✓ セッションを保存しました")
        
        # 読み込み
        loaded = Session.load(save_path)
        assert loaded.session_id == original.session_id
        assert loaded.initial_query == "保存テスト"
        assert len(loaded.query_history) == 1
        assert len(loaded.interpretations) == 1
        assert loaded.selected_interpretation is not None
        assert loaded.selected_interpretation.id == 1
        print("✓ セッションを読み込みました")
        
        # 内容の検証
        assert loaded.session_id == original.session_id
        assert loaded.initial_image_path == original.initial_image_path
        print("✓ 保存・読み込みの内容が一致しました")


def test_session_summary():
    """セッション要約のテスト"""
    print("\n【テスト14】セッション要約")
    
    session = Session()
    session.initial_image_path = "/path/to/image.png"
    session.initial_query = "要約テスト"
    session.add_query("要約テスト")
    session.add_query("詳細化クエリ")
    
    interp1 = Interpretation(id=1, text="解釈1", reasoning="理由1")
    interp2 = Interpretation(id=2, text="解釈2", reasoning="理由2")
    session.add_interpretations([interp1, interp2])
    session.select_interpretation(1)
    
    image = GeneratedImage(image_path="/path/to/gen.png", prompt="Generated")
    session.add_generated_image(image)
    
    constraint = Constraint(
        attribute="attr1",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.5
    )
    session.add_constraint(constraint)
    
    # 要約情報を取得
    summary = session.to_dict()
    
    assert len(summary["query_history"]) == 2
    assert len(summary["interpretations"]) == 2
    assert summary["selected_interpretation"]["id"] == 1
    assert len(summary["generated_images"]) == 1
    assert len(summary["constraints"]) == 1
    
    print("✓ セッション要約情報:")
    print(f"  クエリ数: {len(summary['query_history'])}")
    print(f"  解釈案数: {len(summary['interpretations'])}")
    print(f"  生成画像数: {len(summary['generated_images'])}")
    print(f"  制約数: {len(summary['constraints'])}")


def run_all_tests():
    """全テストを実行"""
    print("=" * 70)
    print("Session 単体テスト")
    print("=" * 70)
    
    tests = [
        test_interpretation_creation,
        test_generated_image_creation,
        test_session_creation,
        test_session_add_query,
        test_session_add_interpretations,
        test_session_select_interpretation,
        test_session_set_vector,
        test_session_add_constraint,
        test_session_add_generated_image,
        test_session_get_latest_image,
        test_session_get_active_constraints,
        test_session_serialization,
        test_session_save_and_load,
        test_session_summary
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ テスト失敗: {test_func.__name__}")
            print(f"  エラー: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ テストエラー: {test_func.__name__}")
            print(f"  例外: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"テスト結果: {passed}件成功, {failed}件失敗")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
