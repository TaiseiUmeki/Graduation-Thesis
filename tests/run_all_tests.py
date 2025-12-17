#!/usr/bin/env python3
"""
全単体テストを実行するスクリプト
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_attribute_space import run_all_tests as test_attribute_space
from tests.test_constraints import run_all_tests as test_constraints
from tests.test_session import run_all_tests as test_session
from tests.test_image_utils import run_all_tests as test_image_utils


def main():
    """全テストを実行"""
    print("\n" + "=" * 70)
    print("TrueCoding システム - 全単体テスト実行")
    print("=" * 70)
    
    test_suites = [
        ("AttributeSpace", test_attribute_space),
        ("Constraints", test_constraints),
        ("Session", test_session),
        ("ImageUtils", test_image_utils),
    ]
    
    results = {}
    
    for suite_name, test_func in test_suites:
        print(f"\n{'='*70}")
        print(f"テストスイート: {suite_name}")
        print(f"{'='*70}")
        
        try:
            success = test_func()
            results[suite_name] = "成功" if success else "失敗"
        except Exception as e:
            print(f"\n✗ テストスイート実行エラー: {e}")
            results[suite_name] = "エラー"
    
    # 最終サマリー
    print("\n" + "=" * 70)
    print("全体サマリー")
    print("=" * 70)
    
    for suite_name, result in results.items():
        status_symbol = "✓" if result == "成功" else "✗"
        print(f"{status_symbol} {suite_name}: {result}")
    
    print("=" * 70)
    
    # 全て成功したか確認
    all_success = all(result == "成功" for result in results.values())
    
    if all_success:
        print("\n✓ 全テスト成功!")
        return 0
    else:
        print("\n✗ 一部のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
