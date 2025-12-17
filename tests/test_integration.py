"""
結合テスト
複数のモジュール間の相互作用をテスト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.attribute_space import ATTR_SPACE, AttributeVector
from models.session import Session, Interpretation, GeneratedImage
from models.constraints import Constraint, ConstraintType, ConstraintManager
import tempfile
import json


def test_session_with_constraints():
    """SessionとConstraintの結合テスト"""
    print("\n【統合テスト1】セッションと制約の連携")
    print("=" * 60)
    
    session = Session()
    print(f"[初期化] セッションID: {session.session_id}")
    
    # 制約を追加
    constraint = Constraint(
        attribute="material:wood_oak",
        constraint_type=ConstraintType.GREATER_THAN,
        value=0.7
    )
    session.add_constraint(constraint)
    
    assert len(session.constraints) == 1
    print(f"[制約追加] 属性: {constraint.attribute}")
    print(f"[制約追加] タイプ: {constraint.constraint_type}")
    print(f"[制約追加] 値: {constraint.value}")
    print("✓ セッションに制約を追加しました")
    
    # ベクトルが制約を満たすか検証
    vector = ATTR_SPACE.create_vector({
        "material:wood_oak": 0.8,
        "color:warm_red": 0.7
    })
    session.set_vector(vector)
    
    print(f"\n[ベクトル作成] 次元数: {len(vector.weights)}")
    print(f"[ベクトル値] material:wood_oak = {vector.weights.get('material:wood_oak', 0):.2f}")
    print(f"[ベクトル値] color:warm_red = {vector.weights.get('color:warm_red', 0):.2f}")
    
    # 制約マネージャーで検証
    mgr = ConstraintManager()
    mgr.add_constraint(constraint)
    is_valid, violations = mgr.validate_vector(vector.weights)
    
    print(f"\n[制約検証] 結果: {'成功' if is_valid else '失敗'}")
    print(f"[制約検証] 違反数: {len(violations)}")
    
    assert is_valid, f"制約違反があります: {violations}"
    print("✓ ベクトルが制約を満たしています")


def test_multiple_constraints():
    """複数の制約の同時検証"""
    print("\n【統合テスト2】複数制約の同時検証")
    print("=" * 60)
    
    mgr = ConstraintManager()
    
    # 複数の制約を設定
    constraints_list = [
        Constraint(
            attribute="material:wood_oak",
            constraint_type=ConstraintType.GREATER_THAN,
            value=0.7
        ),
        Constraint(
            attribute="finish:matte",
            constraint_type=ConstraintType.LESS_THAN,
            value=0.5
        ),
        Constraint(
            attribute="color:warm_red",
            constraint_type=ConstraintType.GREATER_THAN,
            value=0.6
        )
    ]
    
    for i, c in enumerate(constraints_list, 1):
        mgr.add_constraint(c)
        print(f"[制約{i}] {c.attribute} {c.constraint_type.name} {c.value}")
    
    # すべての制約を満たすベクトル
    valid_vector = ATTR_SPACE.create_vector({
        "material:wood_oak": 0.8,
        "finish:matte": 0.3,
        "color:warm_red": 0.65
    })
    
    print(f"\n[ベクトル値]")
    print(f"  material:wood_oak = {valid_vector.weights['material:wood_oak']:.2f} (要件: >= 0.70)")
    print(f"  finish:matte = {valid_vector.weights['finish:matte']:.2f} (要件: < 0.50)")
    print(f"  color:warm_red = {valid_vector.weights['color:warm_red']:.2f} (要件: >= 0.60)")
    
    is_valid, violations = mgr.validate_vector(valid_vector.weights)
    assert is_valid, f"制約検証失敗: {violations}"
    assert len(violations) == 0, f"違反が検出されました: {violations}"
    print(f"\n✓ 全3個の制約を満たすベクトルを検証しました")
    
    # 制約を違反するベクトル
    invalid_vector = ATTR_SPACE.create_vector({
        "material:wood_oak": 0.5,  # 違反
        "finish:matte": 0.3,
        "color:warm_red": 0.7
    })
    
    print(f"\n[違反ベクトル値]")
    print(f"  material:wood_oak = {invalid_vector.weights['material:wood_oak']:.2f} (要件: >= 0.70) ❌")
    print(f"  finish:matte = {invalid_vector.weights['finish:matte']:.2f} (要件: < 0.50) ✓")
    print(f"  color:warm_red = {invalid_vector.weights['color:warm_red']:.2f} (要件: >= 0.60) ✓")
    
    is_valid, violations = mgr.validate_vector(invalid_vector.weights)
    assert not is_valid, f"制約違反が検出されるべき: {is_valid}"
    assert len(violations) > 0, f"違反が検出されていません"
    print(f"\n✓ 制約違反を正しく検出しました")
    print(f"  違反内容: {violations}")


def test_session_persistence_with_constraints():
    """制約付きセッションの永続化"""
    print("\n【統合テスト3】制約付きセッションの保存・読込")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # セッションを作成
        session = Session()
        session.initial_query = "木製で温かみのあるデザイン"
        session.add_query("さらに高級感を出したい")
        
        # ベクトルを追加
        vector = ATTR_SPACE.create_vector({
            "material:wood_oak": 0.8,
            "color:warm_brown": 0.7,
            "finish:matte": 0.8
        })
        session.set_vector(vector)
        
        # 保存
        session.save(tmpdir_path)
        save_path = tmpdir_path / f"{session.session_id}.json"
        assert save_path.exists()
        print(f"✓ セッションを保存しました")
        
        # 読込
        loaded_session = Session.load(save_path)
        
        assert loaded_session.initial_query == session.initial_query, f"初期クエリが一致しません"
        assert len(loaded_session.query_history) == 1, f"クエリ履歴が一致しません: {len(loaded_session.query_history)}"
        assert loaded_session.current_vector is not None, "ベクトルが保存されていません"
        print("✓ セッションを読み込みました")
        print(f"  初期クエリ: {loaded_session.initial_query}")
        print(f"  クエリ履歴: {len(loaded_session.query_history)}件")


def test_attribute_space_with_session():
    """属性空間とセッションの相互作用"""
    print("\n【統合テスト4】属性空間とセッションの連携")
    print("=" * 60)
    
    session = Session()
    print(f"[セッション初期化] ID: {session.session_id}")
    
    # 複合的なベクトルを作成
    weights = {
        "material:wood_oak": 0.9,
        "material:metal_copper": 0.3,
        "finish:matte": 0.8,
        "color:warm_brown": 0.7,
        "style:minimal": 0.6,
        "visual:luxurious": 0.75
    }
    vector = ATTR_SPACE.create_vector(weights)
    
    session.set_vector(vector)
    
    print(f"\n[ベクトル作成] 指定属性数: {len(weights)}")
    print(f"[ベクトル全体] 次元数: {len(vector.weights)}")
    print(f"[指定値一覧]")
    for attr, weight in weights.items():
        print(f"  {ATTR_SPACE.get_attribute_name(attr)}: {weight:.2f}")
    
    # トップ属性を取得
    top_attrs = ATTR_SPACE.get_top_attributes(vector, top_k=3)
    assert len(top_attrs) == 3, f"トップ属性数が異なります: {len(top_attrs)}"
    assert top_attrs[0][0] == "material:wood_oak", f"1位属性が異なります: {top_attrs[0][0]}"
    print(f"\n[トップ3属性]")
    for i, (attr, weight) in enumerate(top_attrs, 1):
        name = ATTR_SPACE.get_attribute_name(attr)
        print(f"  {i}位: {name:20s} {weight:.3f}")
    
    # 自然言語説明を生成
    description = ATTR_SPACE.vector_to_description(vector, threshold=0.5)
    assert "木（オーク）" in description, f"説明に木が含まれていません: {description}"
    print(f"\n[自然言語説明] {description[:70]}...")
    print("✓ 属性空間とセッションの連携が正常です")


def test_vector_merging_in_session():
    """セッション内でのベクトル統合"""
    print("\n【統合テスト5】セッション内でのベクトル統合")
    print("=" * 60)
    
    session = Session()
    
    # 初期ベクトル
    vector1_weights = {
        "material:wood_oak": 0.9,
        "finish:matte": 0.7,
        "color:warm_brown": 0.8
    }
    vector1 = ATTR_SPACE.create_vector(vector1_weights)
    session.set_vector(vector1)
    
    print("[初期ベクトル]")
    for attr, weight in vector1_weights.items():
        print(f"  {ATTR_SPACE.get_attribute_name(attr)}: {weight:.2f}")
    
    # ユーザーフィードバック後のベクトル
    vector2_weights = {
        "material:wood_oak": 0.85,
        "finish:matte": 0.8,
        "color:warm_brown": 0.75,
        "style:minimal": 0.6
    }
    vector2 = ATTR_SPACE.create_vector(vector2_weights)
    
    print("\n[フィードバック後ベクトル]")
    for attr, weight in vector2_weights.items():
        print(f"  {ATTR_SPACE.get_attribute_name(attr)}: {weight:.2f}")
    
    # ベクトルを統合 (alpha=0.6: vector1を60%、vector2を40%)
    alpha = 0.6
    merged = ATTR_SPACE.merge_vectors(vector1, vector2, alpha=alpha)
    
    print(f"\n[統合処理] alpha={alpha} (初期: {100*alpha:.0f}%, フィードバック: {100*(1-alpha):.0f}%)")
    
    # 統合結果を検証
    merged_wood_oak = merged.weights["material:wood_oak"]
    expected_wood_oak = 0.9 * alpha + 0.85 * (1 - alpha)
    
    print(f"\n[統合結果: wood_oak]")
    print(f"  期待値: {expected_wood_oak:.3f}")
    print(f"  実際値: {merged_wood_oak:.3f}")
    
    assert merged.weights["material:wood_oak"] > 0.8
    assert merged.weights["finish:matte"] > 0.7
    print("✓ ベクトルを統合しました")
    print(f"  wood_oak: {merged.weights['material:wood_oak']:.3f}")
    print(f"  matte: {merged.weights['finish:matte']:.3f}")


def test_constraint_guided_workflow():
    """制約ガイド付きのワークフロー"""
    print("\n【統合テスト6】制約ガイド付きワークフロー")
    print("=" * 60)
    
    session = Session()
    session.initial_query = "木製で高級感のあるデザイン"
    print(f"[初期クエリ] {session.initial_query}")
    
    # 制約を設定（要件）
    constraints_list = [
        ("material:wood_oak", ConstraintType.GREATER_THAN, 0.7, "オーク材を優先"),
        ("visual:luxurious", ConstraintType.GREATER_THAN, 0.7, "高級感を出す"),
        ("finish:glossy", ConstraintType.LESS_THAN, 0.3, "光沢を抑える")
    ]
    
    print("\n[制約一覧]")
    for i, (attr, ctype, value, desc) in enumerate(constraints_list, 1):
        c = Constraint(
            attribute=attr,
            constraint_type=ctype,
            value=value,
            description=desc
        )
        session.add_constraint(c)
        print(f"  {i}. {desc}: {attr} {ctype.name} {value}")
    
    # 解釈を生成
    interpretation = Interpretation(
        id=0,
        text="オーク材の自然な質感を活かした上質なデザイン",
        reasoning="高級感とオーク材の要件を両立"
    )
    session.add_interpretations([interpretation])
    session.select_interpretation(0)
    print(f"\n[解釈生成]")
    print(f"  {interpretation.text}")
    
    # 特徴ベクトルを生成（制約を満たす）
    vector_weights = {
        "material:wood_oak": 0.85,
        "visual:luxurious": 0.8,
        "finish:glossy": 0.1,
        "finish:matte": 0.9,
        "color:warm_brown": 0.75
    }
    vector = ATTR_SPACE.create_vector(vector_weights)
    session.set_vector(vector)
    
    print(f"\n[ベクトル値]")
    for attr, weight in vector_weights.items():
        print(f"  {ATTR_SPACE.get_attribute_name(attr)}: {weight:.2f}")
    
    # 制約検証
    mgr = ConstraintManager()
    for constraint in session.get_active_constraints():
        mgr.add_constraint(constraint)
    
    is_valid, violations = mgr.validate_vector(vector.weights)
    assert is_valid, f"制約違反: {violations}"
    print(f"\n[制約検証結果] {'成功 ✓' if is_valid else '失敗 ✗'}")
    print(f"✓ 制約を満たすワークフローを完成させました")
    print(f"  制約数: {len(session.constraints)}")
    print(f"  ベクトル検証: 成功")


def test_multi_iteration_session():
    """複数反復のセッション"""
    print("\n【統合テスト7】複数反復セッション")
    print("=" * 60)
    
    session = Session()
    session.initial_query = "初期クエリ"
    print(f"[初期化] セッションID: {session.session_id}")
    print(f"[初期化] 初期クエリ: {session.initial_query}")
    print(f"[初期化] 初期状態 - クエリ数: 1, 解釈数: 0, 画像数: 0")
    
    # 3回の反復
    for iteration in range(3):
        print(f"\n  【イテレーション {iteration + 1}】")
        print("  " + "-" * 50)
        
        # クエリを更新
        new_query = f"改善クエリ_{iteration + 1}"
        session.add_query(new_query)
        print(f"  [クエリ追加] {new_query}")
        
        # 解釈を追加
        interp = Interpretation(
            id=iteration,
            text=f"解釈_{iteration + 1}",
            reasoning=f"理由_{iteration + 1}"
        )
        session.add_interpretations([interp])
        session.select_interpretation(iteration)
        print(f"  [解釈生成] {interp.text}")
        print(f"  [推論] {interp.reasoning}")
        
        # ベクトルを更新
        base_weight = 0.5 + iteration * 0.1
        weights = {
            "material:wood_oak": base_weight,
            "finish:matte": base_weight + 0.1,
            "style:minimal": max(0, base_weight - 0.1)
        }
        vector = ATTR_SPACE.create_vector(weights)
        session.set_vector(vector)
        
        print(f"  [ベクトル作成] {len(vector.weights)}次元")
        for attr, weight in weights.items():
            print(f"    - {ATTR_SPACE.get_attribute_name(attr)}: {weight:.2f}")
        
        # 画像を追加
        image = GeneratedImage(
            image_path=f"output_{iteration + 1}.png",
            prompt=f"Generated image {iteration + 1}"
        )
        session.add_generated_image(image)
        print(f"  [画像生成] {image.image_path}")
        
        # 累積状態を表示
        print(f"  [累積状態]")
        print(f"    クエリ数: {len(session.query_history)}")
        print(f"    解釈数: {len(session.interpretations)}")
        print(f"    画像数: {len(session.generated_images)}")
    
    # 最終検証
    print(f"\n[最終結果]")
    print(f"✓ {3}回の反復を完了しました")
    print(f"  最終クエリ数: {len(session.query_history)}")
    print(f"  最終解釈数: {len(session.interpretations)}")
    print(f"  最終画像数: {len(session.generated_images)}")
    
    assert len(session.query_history) == 3  # add_queryで追加
    assert len(session.interpretations) == 3
    assert session.current_vector is not None
    assert len(session.generated_images) == 3


def test_end_to_end_workflow():
    """エンドツーエンドの完全ワークフロー (A→B→C→D)"""
    print("\n【統合テスト8】エンドツーエンドワークフロー (A→B→C→D)")
    print("=" * 60)
    
    # Phase A: セッション初期化
    print("\n[フェーズA] セッション初期化")
    print("-" * 60)
    session = Session()
    session.initial_query = "木製で温かみのあるデザイン"
    session.current_phase = "A"
    print(f"  セッションID: {session.session_id}")
    print(f"  初期クエリ: {session.initial_query}")
    print(f"  フェーズ: {session.current_phase}")
    
    # Phase B: 解釈生成
    print("\n[フェーズB] 解釈生成")
    print("-" * 60)
    session.current_phase = "B"
    interpretations = [
        Interpretation(id=0, text="オーク材の温かい色合い", reasoning="自然な木の温もり"),
        Interpretation(id=1, text="無垢材の質感表現", reasoning="加工感をなくす"),
        Interpretation(id=2, text="暖色系の配色", reasoning="温かみを演出")
    ]
    session.add_interpretations(interpretations)
    session.select_interpretation(0)
    print(f"  生成された解釈:")
    for i, interp in enumerate(interpretations, 1):
        mark = "✓" if i == 1 else " "
        print(f"    [{mark}] {i}. {interp.text}")
        print(f"         理由: {interp.reasoning}")
    print(f"  選択解釈: {session.selected_interpretation.text}")
    
    # Phase C: 特徴ベクトル生成
    print("\n[フェーズC] 特徴ベクトル生成")
    print("-" * 60)
    session.current_phase = "C"
    
    vector_weights = {
        "material:wood_oak": 0.9,
        "color:warm_brown": 0.85,
        "finish:matte": 0.7,
        "style:minimal": 0.6,
        "visual:luxurious": 0.75
    }
    vector = ATTR_SPACE.create_vector(vector_weights)
    session.set_vector(vector)
    
    print(f"  ベクトル値 ({len(vector.weights)}次元):")
    for attr, weight in vector_weights.items():
        attr_name = ATTR_SPACE.get_attribute_name(attr)
        if attr_name is None:
            attr_name = attr
        bar = "█" * int(weight * 20)
        print(f"    {str(attr_name):25} {weight:.2f} {bar}")
    
    # 制約を追加
    session.add_constraint(Constraint(
        attribute="finish:glossy",
        constraint_type=ConstraintType.LESS_THAN,
        value=0.2,
        description="光沢を抑える"
    ))
    
    top_attrs = ATTR_SPACE.get_top_attributes(vector, top_k=1)
    constraint_desc = session.constraints[0].to_natural_language()
    print(f"\n  トップ属性: {ATTR_SPACE.get_attribute_name(top_attrs[0][0])} ({top_attrs[0][1]:.2f})")
    print(f"  制約追加: {constraint_desc}")
    
    # Phase D: 画像生成と改善
    print("\n[フェーズD] 画像生成と改善")
    print("-" * 60)
    session.current_phase = "D"
    
    # 初回生成
    image1 = GeneratedImage(
        image_path="output_v1.png",
        prompt="Oak wood with warm brown color and matte finish"
    )
    session.add_generated_image(image1)
    print(f"  【初回生成】")
    print(f"    ファイル: {image1.image_path}")
    print(f"    プロンプト: {image1.prompt}")
    
    # 改善後の生成
    improved_vector = ATTR_SPACE.create_vector({
        "material:wood_oak": 0.95,
        "color:warm_brown": 0.9,
        "finish:matte": 0.8,
        "finish:glossy": 0.05,
        "style:minimal": 0.65,
        "visual:luxurious": 0.85
    })
    session.set_vector(improved_vector)
    
    print(f"\n  【改善ベクトル】(改善量の表示)")
    improvements = {
        "material:wood_oak": (0.9, 0.95),
        "color:warm_brown": (0.85, 0.9),
        "finish:matte": (0.7, 0.8),
        "finish:glossy": (0.0, 0.05),
        "style:minimal": (0.6, 0.65),
        "visual:luxurious": (0.75, 0.85)
    }
    for attr, (old_val, new_val) in improvements.items():
        attr_name = ATTR_SPACE.get_attribute_name(attr)
        if attr_name is None:
            attr_name = attr
        change = new_val - old_val
        change_mark = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"    {str(attr_name):25} {old_val:.2f} → {new_val:.2f} {change_mark} {abs(change):+.2f}")
    
    image2 = GeneratedImage(
        image_path="output_v2.png",
        prompt="Oak wood, warm brown, matte finish, high-end look"
    )
    session.add_generated_image(image2)
    print(f"\n  【改善画像生成】")
    print(f"    ファイル: {image2.image_path}")
    print(f"    プロンプト: {image2.prompt}")
    
    # 最終検証
    print(f"\n[最終結果]")
    print(f"  セッション状態:")
    print(f"    フェーズ: {session.current_phase}")
    print(f"    解釈案数: {len(session.interpretations)}")
    print(f"    生成画像数: {len(session.generated_images)}")
    print(f"    制約数: {len(session.constraints)}")
    print(f"\n✓ エンドツーエンドワークフロー (A→B→C→D) を完了しました")
    
    assert session.current_phase == "D"
    assert len(session.interpretations) == 3
    assert len(session.generated_images) == 2
    assert len(session.constraints) == 1


def run_all_integration_tests():
    """全結合テストを実行"""
    print("=" * 70)
    print("結合テスト実行")
    print("=" * 70)
    
    tests = [
        test_session_with_constraints,
        test_multiple_constraints,
        test_session_persistence_with_constraints,
        test_attribute_space_with_session,
        test_vector_merging_in_session,
        test_constraint_guided_workflow,
        test_multi_iteration_session,
        test_end_to_end_workflow,
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
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
