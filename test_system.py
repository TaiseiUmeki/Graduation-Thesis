"""
システムの基本動作テスト
"""
import os
import sys
from pathlib import Path

# 環境変数を読み込む
from dotenv import load_dotenv
load_dotenv()

# APIキーの確認
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "your-api-key-here":
    print("❌ エラー: OPENAI_API_KEYが設定されていません")
    print("   .envファイルを編集してAPIキーを設定してください")
    sys.exit(1)

print("✅ APIキーが設定されました")

# システムのインポート確認
try:
    from main import TrueCodingSystem
    from config import Config
    from models.attribute_space import ATTR_SPACE
    print("✅ すべてのモジュールが正常にインポートされました")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)

# ディレクトリ構造の確認
def check_directories():
    """必要なディレクトリが存在するか確認"""
    dirs = [
        Config.SESSIONS_DIR,
        Config.IMAGES_DIR,
        Config.OUTPUTS_DIR
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")

print("\n【ディレクトリ確認】")
check_directories()

# 属性空間の確認
print("\n【属性空間の確認】")
print(f"総属性数: {len(ATTR_SPACE.all_attributes)}")
print(f"グループ数: {len(ATTR_SPACE.groups)}")
print("\nグループ一覧:")
for group_name in ATTR_SPACE.groups.keys():
    count = len(ATTR_SPACE.groups[group_name])
    print(f"  - {group_name}: {count}個の属性")

# システム初期化テスト
print("\n【システム初期化テスト】")
try:
    system = TrueCodingSystem()
    print("✅ TrueCodingSystemが正常に初期化されました")
except Exception as e:
    print(f"❌ システム初期化エラー: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ すべてのテストに合格しました！")
print("=" * 60)
print("\n次のステップ:")
print("1. テスト画像を用意: ~/university/lab_research/truecoding/test_image.jpg")
print("2. 以下のコマンドでシステムを実行:")
print("   python examples.py")