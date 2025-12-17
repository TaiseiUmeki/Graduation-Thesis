"""
attribute_space.py の単体テスト
APIを使用しないため、すぐに実行可能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.attribute_space import AttributeSpace, AttributeVector, ATTR_SPACE


def test_attribute_space_initialization():
    """属性空間の初期化テスト"""
    print("\n【テスト1】属性空間の初期化")
    
    assert ATTR_SPACE is not None, "ATTR_SPACE が初期化されていません"
    assert len(ATTR_SPACE.groups) > 0, "グループが存在しません"
    assert len(ATTR_SPACE.all_attributes) > 0, "属性が存在しません"
    
    print(f"✓ グループ数: {len(ATTR_SPACE.groups)}")
    print(f"✓ 総属性数: {len(ATTR_SPACE.all_attributes)}")
    
    # 各グループの存在確認
    expected_groups = ['material', 'finish', 'shape', 'size', 'color', 
                       'texture', 'structure', 'function', 'style', 'visual', 'pattern']
    for group in expected_groups:
        assert group in ATTR_SPACE.groups, f"グループ {group} が見つかりません"
    print(f"✓ 全{len(expected_groups)}グループが存在します")


def test_zero_vector():
    """ゼロベクトルの生成テスト"""
    print("\n【テスト2】ゼロベクトルの生成")
    
    vector = ATTR_SPACE.zero_vector()
    
    assert isinstance(vector, AttributeVector), "AttributeVectorインスタンスではありません"
    assert len(vector.weights) == len(ATTR_SPACE.all_attributes), "属性数が一致しません"
    assert all(w == 0.0 for w in vector.weights.values()), "ゼロでない要素があります"
    
    print(f"✓ {len(vector.weights)}個の属性がすべて0.0で初期化されました")


def test_create_vector():
    """カスタムベクトル生成のテスト"""
    print("\n【テスト3】カスタムベクトルの生成")
    
    weights = {
        "material:wood_oak": 0.8,
        "finish:matte": 0.7,
        "style:minimal": 0.9
    }
    
    vector = ATTR_SPACE.create_vector(weights)
    
    assert vector.weights["material:wood_oak"] == 0.8
    assert vector.weights["finish:matte"] == 0.7
    assert vector.weights["style:minimal"] == 0.9
    
    print("✓ 指定した重みでベクトルが生成されました")
    print(f"  material:wood_oak = {vector.weights['material:wood_oak']}")
    print(f"  finish:matte = {vector.weights['finish:matte']}")
    print(f"  style:minimal = {vector.weights['style:minimal']}")
    
    # 範囲外の値のクリッピングテスト
    invalid_weights = {
        "material:stone": 1.5,  # 1.0を超える
        "color:warm_red": -0.3  # 0.0未満
    }
    vector2 = ATTR_SPACE.create_vector(invalid_weights)
    
    assert vector2.weights["material:stone"] == 1.0, "上限クリッピングが機能していません"
    assert vector2.weights["color:warm_red"] == 0.0, "下限クリッピングが機能していません"
    print("✓ 範囲外の値が正しくクリッピングされました")


def test_get_attribute_name():
    """属性名の取得テスト"""
    print("\n【テスト4】属性名の取得")
    
    name = ATTR_SPACE.get_attribute_name("material:wood_oak")
    assert name == "木（オーク）", f"属性名が正しくありません: {name}"
    print(f"✓ material:wood_oak → {name}")
    
    name2 = ATTR_SPACE.get_attribute_name("finish:matte")
    assert name2 == "マット", f"属性名が正しくありません: {name2}"
    print(f"✓ finish:matte → {name2}")
    
    # 存在しない属性
    invalid_name = ATTR_SPACE.get_attribute_name("invalid:attribute")
    assert invalid_name is None, "存在しない属性でNoneが返されませんでした"
    print("✓ 存在しない属性に対してNoneを返します")


def test_get_top_attributes():
    """上位属性の取得テスト"""
    print("\n【テスト5】上位属性の取得")
    
    weights = {
        "material:wood_oak": 0.9,
        "finish:matte": 0.7,
        "style:minimal": 0.85,
        "color:warm_brown": 0.6,
        "texture:smooth": 0.5
    }
    vector = ATTR_SPACE.create_vector(weights)
    
    top_attrs = ATTR_SPACE.get_top_attributes(vector, top_k=3)
    
    assert len(top_attrs) == 3, f"上位3件を取得できませんでした: {len(top_attrs)}"
    assert top_attrs[0][0] == "material:wood_oak", "1位が正しくありません"
    assert top_attrs[1][0] == "style:minimal", "2位が正しくありません"
    assert top_attrs[2][0] == "finish:matte", "3位が正しくありません"
    
    print("✓ 上位3属性:")
    for i, (attr, weight) in enumerate(top_attrs, 1):
        name = ATTR_SPACE.get_attribute_name(attr)
        print(f"  {i}位: {name} ({attr}) = {weight}")


def test_vector_to_description():
    """ベクトルの自然言語変換テスト"""
    print("\n【テスト6】ベクトルの自然言語変換")
    
    weights = {
        "material:wood_oak": 0.8,
        "finish:matte": 0.7,
        "style:minimal": 0.6
    }
    vector = ATTR_SPACE.create_vector(weights)
    
    # 閾値0.5で変換
    description = ATTR_SPACE.vector_to_description(vector, threshold=0.5)
    
    assert "木（オーク）" in description, "木（オーク）が含まれていません"
    assert "マット" in description, "マットが含まれていません"
    assert "ミニマル" in description, "ミニマルが含まれていません"
    
    print(f"✓ 生成された説明:\n  {description}")
    
    # 閾値を上げた場合
    description2 = ATTR_SPACE.vector_to_description(vector, threshold=0.75)
    assert "木（オーク）" in description2, "閾値を上げても木（オーク）は含まれるべきです"
    print(f"✓ 閾値0.75での説明:\n  {description2}")


def test_merge_vectors():
    """ベクトルの統合テスト"""
    print("\n【テスト7】ベクトルの統合")
    
    vector1 = ATTR_SPACE.create_vector({
        "material:wood_oak": 0.8,
        "finish:matte": 0.6
    })
    
    vector2 = ATTR_SPACE.create_vector({
        "material:wood_oak": 0.4,
        "finish:glossy": 0.7
    })
    
    # alpha=0.5 で均等に統合
    merged = ATTR_SPACE.merge_vectors(vector1, vector2, alpha=0.5)
    
    expected_oak = 0.5 * 0.8 + 0.5 * 0.4  # 0.6
    assert abs(merged.weights["material:wood_oak"] - expected_oak) < 0.01, \
        f"統合値が正しくありません: {merged.weights['material:wood_oak']}"
    
    print(f"✓ vector1の material:wood_oak = 0.8")
    print(f"✓ vector2の material:wood_oak = 0.4")
    print(f"✓ 統合後 (alpha=0.5) = {merged.weights['material:wood_oak']:.2f}")
    
    # alpha=0.7 で vector1 を優先
    merged2 = ATTR_SPACE.merge_vectors(vector1, vector2, alpha=0.7)
    expected_oak2 = 0.7 * 0.8 + 0.3 * 0.4  # 0.68
    assert abs(merged2.weights["material:wood_oak"] - expected_oak2) < 0.01
    print(f"✓ 統合後 (alpha=0.7) = {merged2.weights['material:wood_oak']:.2f}")


def test_vector_serialization():
    """ベクトルのシリアライゼーションテスト"""
    print("\n【テスト8】ベクトルのシリアライゼーション")
    
    weights = {
        "material:wood_oak": 0.8,
        "finish:matte": 0.7
    }
    original = AttributeVector(weights=weights)
    
    # 辞書に変換
    dict_data = original.to_dict()
    assert "weights" in dict_data
    assert dict_data["weights"]["material:wood_oak"] == 0.8
    print("✓ 辞書への変換成功")
    
    # JSONに変換
    json_str = original.to_json()
    assert "material:wood_oak" in json_str
    print("✓ JSONへの変換成功")
    
    # JSONから復元
    restored = AttributeVector.from_json(json_str)
    assert restored.weights["material:wood_oak"] == 0.8
    assert restored.weights["finish:matte"] == 0.7
    print("✓ JSONからの復元成功")
    
    # 辞書から復元
    restored2 = AttributeVector.from_dict(dict_data)
    assert restored2.weights["material:wood_oak"] == 0.8
    print("✓ 辞書からの復元成功")


def test_get_group_attributes():
    """グループ別属性取得テスト"""
    print("\n【テスト9】グループ別属性取得")
    
    material_attrs = ATTR_SPACE.get_group_attributes("material")
    assert len(material_attrs) > 0, "material グループが空です"
    assert "wood_oak" in material_attrs, "wood_oak が material グループにありません"
    print(f"✓ material グループ: {len(material_attrs)}個の属性")
    
    # いくつかの属性を表示
    for key, name in list(material_attrs.items())[:5]:
        print(f"  - {key}: {name}")
    
    # 存在しないグループ
    invalid_group = ATTR_SPACE.get_group_attributes("nonexistent")
    assert len(invalid_group) == 0, "存在しないグループで空辞書が返されませんでした"
    print("✓ 存在しないグループに対して空辞書を返します")


def run_all_tests():
    """全テストを実行"""
    print("=" * 70)
    print("AttributeSpace 単体テスト")
    print("=" * 70)
    
    tests = [
        test_attribute_space_initialization,
        test_zero_vector,
        test_create_vector,
        test_get_attribute_name,
        test_get_top_attributes,
        test_vector_to_description,
        test_merge_vectors,
        test_vector_serialization,
        test_get_group_attributes
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
