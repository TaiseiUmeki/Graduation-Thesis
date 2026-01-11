"""
画像ユーティリティモジュール
画像の読み込み、保存、変換などの処理
"""
from pathlib import Path
from typing import Optional, Tuple
import shutil
from datetime import datetime
import uuid

import requests
from PIL import Image
import base64
from io import BytesIO


class ImageUtils:
    """画像処理ユーティリティクラス"""
    
    @staticmethod
    def save_uploaded_image(
        image_path: str,
        output_dir: Path,
        prefix: str = "input"
    ) -> str:
        """
        アップロードされた画像を保存
        
        Args:
            image_path: 元の画像パス
            output_dir: 出力ディレクトリ
            prefix: ファイル名の接頭辞
        
        Returns:
            保存先のパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 拡張子を取得
        ext = Path(image_path).suffix
        if not ext:
            ext = '.png'
        
        # 一意なファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}{ext}"
        
        output_path = output_dir / filename
        shutil.copy(image_path, output_path)
        
        return str(output_path)
    
    @staticmethod
    def download_image_from_url(
        url: str,
        output_dir: Path,
        prefix: str = "generated"
    ) -> str:
        """
        URLから画像をダウンロードして保存
        
        Args:
            url: 画像のURL
            output_dir: 出力ディレクトリ
            prefix: ファイル名の接頭辞
        
        Returns:
            保存先のパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像をダウンロード
        response = requests.get(url)
        response.raise_for_status()
        
        # 一意なファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.png"
        
        output_path = output_dir / filename
        
        # 画像を保存
        img = Image.open(BytesIO(response.content))
        img.save(output_path)
        
        return str(output_path)
    
    @staticmethod
    def resize_image(
        image_path: str,
        max_size: Tuple[int, int] = (1024, 1024),
        output_path: Optional[str] = None
    ) -> str:
        """
        画像をリサイズ
        
        Args:
            image_path: 入力画像パス
            max_size: 最大サイズ（幅, 高さ）
            output_path: 出力パス（Noneの場合は上書き）
        
        Returns:
            リサイズ後の画像パス
        """
        img = Image.open(image_path)
        
        # アスペクト比を保持してリサイズ
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        if output_path is None:
            output_path = image_path
        
        img.save(output_path)
        return output_path
    
    @staticmethod
    def convert_to_rgb(image_path: str, output_path: Optional[str] = None) -> str:
        """
        画像をRGBモードに変換
        
        Args:
            image_path: 入力画像パス
            output_path: 出力パス（Noneの場合は上書き）
        
        Returns:
            変換後の画像パス
        """
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if output_path is None:
            output_path = image_path
        
        img.save(output_path)
        return output_path
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """
        画像をBase64文字列に変換
        
        Args:
            image_path: 画像パス
        
        Returns:
            Base64エンコードされた文字列
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def base64_to_image(base64_str: str, output_path: str) -> str:
        """
        Base64文字列を画像に変換
        
        Args:
            base64_str: Base64文字列
            output_path: 出力パス
        
        Returns:
            保存した画像のパス
        """
        image_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(image_data))
        img.save(output_path)
        return output_path

    @staticmethod
    def save_base64_image(
        base64_str: str,
        output_dir: Path,
        prefix: str = "generated"
    ) -> str:
        """
        Base64画像をファイルとして保存
        
        Args:
            base64_str: Base64文字列
            output_dir: 出力ディレクトリ
            prefix: ファイル名の接頭辞
        
        Returns:
            保存先のパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.png"
        
        output_path = output_dir / filename
        return ImageUtils.base64_to_image(base64_str, str(output_path))
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """
        画像の情報を取得
        
        Args:
            image_path: 画像パス
        
        Returns:
            画像情報の辞書
        """
        img = Image.open(image_path)
        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
            "size_bytes": Path(image_path).stat().st_size
        }
    
    @staticmethod
    def create_thumbnail(
        image_path: str,
        size: Tuple[int, int] = (256, 256),
        output_dir: Optional[Path] = None
    ) -> str:
        """
        サムネイル画像を作成
        
        Args:
            image_path: 入力画像パス
            size: サムネイルサイズ
            output_dir: 出力ディレクトリ（Noneの場合は元画像と同じ場所）
        
        Returns:
            サムネイル画像のパス
        """
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # 出力パスを決定
        path = Path(image_path)
        if output_dir is None:
            output_dir = path.parent
        
        output_path = output_dir / f"{path.stem}_thumb{path.suffix}"
        img.save(output_path)
        
        return str(output_path)
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        画像ファイルが有効かどうかチェック
        
        Args:
            image_path: 画像パス
        
        Returns:
            有効ならTrue
        """
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception:
            return False
