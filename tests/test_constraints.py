"""
constraints.py の単体テスト
制約の作成、検証、管理機能をテスト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.constraints import Constraint, ConstraintType, ConstraintManager
from datetime import datetime


def test_constraint_creation():
    """制約の作成テスト"""
    print("\n【テスト1】制約の作成")
    
    # LESS_THAN 制約
    c1 = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5,
        description="オーク材を控えめに"
    )
    
    assert c1.attribute == "material:wood_oak"
    assert c1.constraint_type == ConstraintType.LESS_THAN
    assert c1.value == 0.5
    assert c1.is_active is True
    print("✓ LESS_THAN 制約を作成")
    
    # GREATER_THAN 制約
    c2 = Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7,
        description="マット仕上げを強調"
    )
    print("✓ GREATER_THAN 制約を作成")
    
    # RANGE 制約
    c3 = Constraint(
        attribute="size:medium",
        constraint_type=ConstraintType.RANGE,
        min_value=0.4,
        max_value=0.8,
        description="中サイズの範囲"
    )
    print("✓ RANGE 制約を作成")
    
    # タイムスタンプの確認
    assert isinstance(c1.timestamp, datetime)
    print("✓ タイムスタンプが自動設定されました")


def test_constraint_validation():
    """制約の検証テスト"""
    print("\n【テスト2】制約の検証")
    
    # LESS_THAN テスト
    c1 = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5
    )
    
    assert c1.validate(0.3) is True, "0.3 <= 0.5 は True であるべき"
    assert c1.validate(0.5) is True, "0.5 <= 0.5 は True であるべき"
    assert c1.validate(0.7) is False, "0.7 <= 0.5 は False であるべき"
    print("✓ LESS_THAN の検証が正しい")
    
    # GREATER_THAN テスト
    c2 = Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.6
    )
    
    assert c2.validate(0.8) is True, "0.8 >= 0.6 は True であるべき"
    assert c2.validate(0.6) is True, "0.6 >= 0.6 は True であるべき"
    assert c2.validate(0.4) is False, "0.4 >= 0.6 は False であるべき"
    print("✓ GREATER_THAN の検証が正しい")
    
    # EQUAL テスト
    c3 = Constraint(
        attribute="size:medium",
        constraint_type=ConstraintType.EQUAL,
        value=0.5
    )
    
    assert c3.validate(0.5) is True, "0.5 == 0.5 は True であるべき"
    assert c3.validate(0.50000001) is True, "微小な誤差は許容すべき"
    assert c3.validate(0.6) is False, "0.6 == 0.5 は False であるべき"
    print("✓ EQUAL の検証が正しい")
    
    # RANGE テスト
    c4 = Constraint(
        attribute="color:saturation",
        constraint_type=ConstraintType.RANGE,
        min_value=0.3,
        max_value=0.7
    )
    
    assert c4.validate(0.5) is True, "0.3 <= 0.5 <= 0.7 は True"
    assert c4.validate(0.3) is True, "境界値 0.3 は True"
    assert c4.validate(0.7) is True, "境界値 0.7 は True"
    assert c4.validate(0.2) is False, "0.2 < 0.3 は False"
    assert c4.validate(0.8) is False, "0.8 > 0.7 は False"
    print("✓ RANGE の検証が正しい")


def test_inactive_constraint():
    """非アクティブな制約のテスト"""
    print("\n【テスト3】非アクティブな制約")
    
    c = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5,
        is_active=False
    )
    
    # 非アクティブな制約は常にTrueを返すべき
    assert c.validate(1.0) is True, "非アクティブな制約は常にTrueであるべき"
    assert c.validate(0.0) is True
    print("✓ 非アクティブな制約は全ての値を許容します")


def test_constraint_natural_language():
    """制約の自然言語変換テスト"""
    print("\n【テスト4】制約の自然言語変換")
    
    c1 = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5
    )
    nl1 = c1.to_natural_language()
    assert "material:wood_oak" in nl1
    assert "0.5" in nl1 or "0.50" in nl1
    assert "≤" in nl1
    print(f"✓ LESS_THAN: {nl1}")
    
    c2 = Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7
    )
    nl2 = c2.to_natural_language()
    assert "≥" in nl2
    print(f"✓ GREATER_THAN: {nl2}")
    
    c3 = Constraint(
        attribute="size:medium",
        constraint_type=ConstraintType.RANGE,
        min_value=0.4,
        max_value=0.8
    )
    nl3 = c3.to_natural_language()
    assert "0.4" in nl3 or "0.40" in nl3
    assert "0.8" in nl3 or "0.80" in nl3
    print(f"✓ RANGE: {nl3}")


def test_constraint_serialization():
    """制約のシリアライゼーションテスト"""
    print("\n【テスト5】制約のシリアライゼーション")
    
    original = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5,
        description="テスト制約"
    )
    
    # 辞書に変換
    dict_data = original.to_dict()
    assert dict_data["attribute"] == "material:wood_oak"
    assert dict_data["constraint_type"] == "less_than"
    assert dict_data["value"] == 0.5
    assert dict_data["description"] == "テスト制約"
    print("✓ 辞書への変換成功")
    
    # 辞書から復元
    restored = Constraint.from_dict(dict_data)
    assert restored.attribute == original.attribute
    assert restored.constraint_type == original.constraint_type
    assert restored.value == original.value
    assert restored.description == original.description
    print("✓ 辞書からの復元成功")


def test_constraint_manager_basic():
    """ConstraintManagerの基本機能テスト"""
    print("\n【テスト6】ConstraintManagerの基本機能")
    
    manager = ConstraintManager()
    
    # 制約を追加
    c1 = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7
    )
    manager.add_constraint(c1)
    
    c2 = Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5
    )
    manager.add_constraint(c2)
    
    assert len(manager.constraints) == 2, "2つの制約が追加されているべき"
    print("✓ 制約の追加成功")
    
    # アクティブな制約の取得
    active = manager.get_active_constraints()
    assert len(active) == 2, "全ての制約がアクティブであるべき"
    print("✓ アクティブな制約の取得成功")
    
    # 制約の無効化
    manager.deactivate_constraint(0)
    active = manager.get_active_constraints()
    assert len(active) == 1, "1つの制約のみアクティブであるべき"
    print("✓ 制約の無効化成功")
    
    # 制約の再有効化
    manager.activate_constraint(0)
    active = manager.get_active_constraints()
    assert len(active) == 2, "再び2つの制約がアクティブであるべき"
    print("✓ 制約の再有効化成功")
    
    # 制約の削除
    manager.remove_constraint(0)
    assert len(manager.constraints) == 1, "1つの制約が残っているべき"
    print("✓ 制約の削除成功")


def test_constraint_manager_validation():
    """ConstraintManagerの検証機能テスト"""
    print("\n【テスト7】ConstraintManagerの検証機能")
    
    manager = ConstraintManager()
    
    # 制約を追加
    manager.add_constraint(Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7,
        description="オーク材を強調"
    ))
    
    manager.add_constraint(Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.5,
        description="マットを控えめに"
    ))
    
    # 制約を満たすベクトル
    valid_vector = {
        "material:wood_oak": 0.8,
        "finish:matte": 0.4
    }
    is_valid, violations = manager.validate_vector(valid_vector)
    assert is_valid is True, "このベクトルは制約を満たすべき"
    assert len(violations) == 0, "違反がないはず"
    print("✓ 制約を満たすベクトルを正しく検証")
    
    # 制約を満たさないベクトル
    invalid_vector = {
        "material:wood_oak": 0.5,  # 0.7以上が必要
        "finish:matte": 0.6         # 0.5以下が必要
    }
    is_valid, violations = manager.validate_vector(invalid_vector)
    assert is_valid is False, "このベクトルは制約を満たさないはず"
    assert len(violations) == 2, "2つの違反があるはず"
    print("✓ 制約違反を正しく検出")
    print(f"  違反内容: {violations}")
    
    # 一部制約を満たさないベクトル
    partial_invalid = {
        "material:wood_oak": 0.8,  # OK
        "finish:matte": 0.7         # NG
    }
    is_valid, violations = manager.validate_vector(partial_invalid)
    assert is_valid is False
    assert len(violations) == 1
    print("✓ 一部の制約違反を正しく検出")


def test_constraint_manager_for_attribute():
    """特定属性の制約取得テスト"""
    print("\n【テスト8】特定属性の制約取得")
    
    manager = ConstraintManager()
    
    manager.add_constraint(Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7
    ))
    
    manager.add_constraint(Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.9
    ))
    
    manager.add_constraint(Constraint(
        attribute="finish:matte",
        constraint_type=ConstraintType.EQUAL,
        value=0.5
    ))
    
    # wood_oak の制約を取得
    oak_constraints = manager.get_constraints_for_attribute("material:wood_oak")
    assert len(oak_constraints) == 2, "wood_oakに2つの制約があるはず"
    print(f"✓ material:wood_oak に {len(oak_constraints)} 個の制約")
    
    # matte の制約を取得
    matte_constraints = manager.get_constraints_for_attribute("finish:matte")
    assert len(matte_constraints) == 1, "matteに1つの制約があるはず"
    print(f"✓ finish:matte に {len(matte_constraints)} 個の制約")
    
    # 存在しない属性
    none_constraints = manager.get_constraints_for_attribute("nonexistent:attr")
    assert len(none_constraints) == 0, "存在しない属性には制約がないはず"
    print("✓ 存在しない属性に対して空リストを返す")


def test_constraint_manager_to_prompt():
    """制約のプロンプト変換テスト"""
    print("\n【テスト9】制約のプロンプト変換")
    
    manager = ConstraintManager()
    
    # 制約がない場合
    prompt1 = manager.to_prompt_text()
    assert "制約なし" in prompt1
    print("✓ 制約なしの場合の出力")
    
    # 制約を追加
    manager.add_constraint(Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.8,
        description="オーク材の質感を強調する"
    ))
    
    manager.add_constraint(Constraint(
        attribute="finish:glossy",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.2,
        description="光沢を抑える"
    ))
    
    prompt2 = manager.to_prompt_text()
    assert "以下の制約を満たす" in prompt2
    assert "オーク材" in prompt2
    assert "光沢" in prompt2
    print("✓ 制約ありの場合の出力:")
    print(prompt2)


def run_all_tests():
    """全テストを実行"""
    print("=" * 70)
    print("Constraints 単体テスト")
    print("=" * 70)
    
    tests = [
        test_constraint_creation,
        test_constraint_validation,
        test_inactive_constraint,
        test_constraint_natural_language,
        test_constraint_serialization,
        test_constraint_manager_basic,
        test_constraint_manager_validation,
        test_constraint_manager_for_attribute,
        test_constraint_manager_to_prompt
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
