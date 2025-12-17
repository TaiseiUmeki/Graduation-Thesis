"""
image_utils.py の単体テスト
画像ユーティリティ機能のテスト(APIを使用しない部分のみ)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import ImageUtils
from PIL import Image
import tempfile
import base64
import shutil


def test_save_uploaded_image():
    """アップロード画像保存のテスト"""
    print("\n【テスト1】アップロード画像の保存")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # テスト用の元画像を作成
        source_image = Image.new('RGB', (100, 100), color='red')
        source_path = tmpdir_path / "source.png"
        source_image.save(source_path)
        
        # 画像を保存
        output_dir = tmpdir_path / "output"
        save_path = ImageUtils.save_uploaded_image(str(source_path), output_dir, prefix="test")
        
        assert Path(save_path).exists(), "画像ファイルが作成されていません"
        assert "test_" in Path(save_path).name
        print(f"✓ 画像を保存しました: {Path(save_path).name}")
        
        # 保存された画像を読み込んで確認
        loaded_image = Image.open(save_path)
        assert loaded_image.size == (100, 100)
        print("✓ 保存した画像を読み込めました")


def test_get_image_info():
    """画像情報取得のテスト"""
    print("\n【テスト2】画像情報の取得")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # テスト画像を作成
        test_image = Image.new('RGB', (640, 480), color='blue')
        image_path = tmpdir_path / "test.png"
        test_image.save(image_path)
        
        # 画像情報を取得
        info = ImageUtils.get_image_info(str(image_path))
        
        assert info["format"] == "PNG"
        assert info["width"] == 640
        assert info["height"] == 480
        assert info["mode"] == "RGB"
        
        print(f"✓ 画像情報を取得しました:")
        print(f"  フォーマット: {info['format']}")
        print(f"  サイズ: {info['width']}x{info['height']}")
        print(f"  モード: {info['mode']}")


def test_validate_image():
    """画像検証のテスト"""
    print("\n【テスト3】画像の検証")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # 有効な画像
        valid_image = Image.new('RGB', (50, 50), color='green')
        valid_path = tmpdir_path / "valid.png"
        valid_image.save(valid_path)
        
        assert ImageUtils.validate_image(str(valid_path)), "有効な画像が無効と判定されました"
        print("✓ 有効な画像を正しく検証しました")
        
        # 無効な画像（テキストファイル）
        invalid_path = tmpdir_path / "invalid.png"
        invalid_path.write_text("not an image")
        
        assert not ImageUtils.validate_image(str(invalid_path)), "無効な画像が有効と判定されました"
        print("✓ 無効な画像を正しく検出しました")


def test_image_to_base64():
    """画像→Base64変換のテスト"""
    print("\n【テスト4】画像→Base64変換")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # テスト用の画像を作成
        test_image = Image.new('RGB', (10, 10), color='yellow')
        image_path = tmpdir_path / "test.png"
        test_image.save(image_path)
        
        # Base64に変換
        base64_str = ImageUtils.image_to_base64(str(image_path))
        
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        print(f"✓ Base64文字列を生成しました (長さ: {len(base64_str)}文字)")
        
        # Base64から画像に戻す
        output_path = tmpdir_path / "decoded.png"
        ImageUtils.base64_to_image(base64_str, str(output_path))
        
        decoded_image = Image.open(output_path)
        assert decoded_image.size == (10, 10)
        print("✓ Base64から画像に復元できました")


def test_create_thumbnail():
    """サムネイル作成のテスト"""
    print("\n【テスト5】サムネイルの作成")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # 大きな画像を作成
        large_image = Image.new('RGB', (1000, 800), color='cyan')
        image_path = tmpdir_path / "large.png"
        large_image.save(image_path)
        
        # サムネイルを作成
        thumb_path = ImageUtils.create_thumbnail(str(image_path), size=(200, 200))
        
        thumbnail = Image.open(thumb_path)
        assert thumbnail.size[0] <= 200
        assert thumbnail.size[1] <= 200
        print(f"✓ サムネイルを作成しました: {thumbnail.size}")
        
        # アスペクト比が保たれているか確認
        original_aspect = large_image.size[0] / large_image.size[1]
        thumbnail_aspect = thumbnail.size[0] / thumbnail.size[1]
        
        assert abs(original_aspect - thumbnail_aspect) < 0.01, "アスペクト比が保たれていません"
        print("✓ アスペクト比が保たれています")


def test_resize_image():
    """画像リサイズのテスト"""
    print("\n【テスト6】画像のリサイズ")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # オリジナル画像を作成
        original = Image.new('RGB', (800, 600), color='pink')
        image_path = tmpdir_path / "original.png"
        original.save(image_path)
        
        # リサイズ
        resized_path = ImageUtils.resize_image(str(image_path), max_size=(400, 300))
        
        resized = Image.open(resized_path)
        assert resized.size[0] <= 400
        assert resized.size[1] <= 300
        print(f"✓ 画像をリサイズしました: {original.size} → {resized.size}")


def test_convert_to_rgb():
    """RGB変換のテスト"""
    print("\n【テスト7】RGB変換")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # RGBA画像を作成
        rgba_image = Image.new('RGBA', (50, 50), color=(255, 0, 0, 128))
        rgba_path = tmpdir_path / "rgba.png"
        rgba_image.save(rgba_path)
        
        # RGBに変換
        rgb_path = ImageUtils.convert_to_rgb(str(rgba_path))
        
        rgb_image = Image.open(rgb_path)
        assert rgb_image.mode == 'RGB'
        assert rgb_image.size == (50, 50)
        print(f"✓ RGBAからRGBに変換しました")
        
        # すでにRGBの場合
        rgb_original = Image.new('RGB', (50, 50), color='blue')
        rgb_orig_path = tmpdir_path / "rgb_orig.png"
        rgb_original.save(rgb_orig_path)
        
        rgb_result_path = ImageUtils.convert_to_rgb(str(rgb_orig_path))
        rgb_result = Image.open(rgb_result_path)
        
        assert rgb_result.mode == 'RGB'
        print("✓ RGB画像はそのまま保持されました")


def run_all_tests():
    """全テストを実行"""
    print("=" * 70)
    print("ImageUtils 単体テスト")
    print("=" * 70)
    
    tests = [
        test_save_uploaded_image,
        test_get_image_info,
        test_validate_image,
        test_image_to_base64,
        test_create_thumbnail,
        test_resize_image,
        test_convert_to_rgb
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
