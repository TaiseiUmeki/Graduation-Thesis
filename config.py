"""
設定管理モジュール
OpenAI APIキーや各種パラメータを管理
"""
import os
from pathlib import Path
from typing import Optional

class Config:
    """システム設定クラス"""
    
    # ベースディレクトリ
    BASE_DIR = Path(__file__).parent
    
    # データディレクトリ
    DATA_DIR = BASE_DIR / "data"
    SESSIONS_DIR = DATA_DIR / "sessions"
    IMAGES_DIR = DATA_DIR / "images"
    OUTPUTS_DIR = DATA_DIR / "outputs"
    
    # OpenAI API設定
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # モデル設定
    GPT_MODEL = "gpt-4o"  # クエリ解釈・特徴ベクトル生成用
    VISION_MODEL = "gpt-4o"  # 画像理解用
    IMAGE_MODEL = "dall-e-3"  # 画像生成用
    
    # パラメータ設定
    TEMPERATURE = 0.7  # 生成の多様性（0.0-2.0）
    MAX_TOKENS = 2000  # 最大トークン数
    NUM_INTERPRETATIONS = 3  # 解釈案の数
    
    # 画像生成設定
    IMAGE_SIZE = "1024x1024"  # 生成画像サイズ
    IMAGE_QUALITY = "standard"  # standard or hd
    
    @classmethod
    def ensure_directories(cls):
        """必要なディレクトリを作成"""
        for dir_path in [cls.DATA_DIR, cls.SESSIONS_DIR, cls.IMAGES_DIR, cls.OUTPUTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if not cls.OPENAI_API_KEY:
            print("警告: OPENAI_API_KEYが設定されていません")
            print("環境変数に設定するか、.envファイルを使用してください")
    
    @classmethod
    def set_api_key(cls, api_key: str):
        """APIキーを設定"""
        cls.OPENAI_API_KEY = api_key
        os.environ["OPENAI_API_KEY"] = api_key


# 初期化
Config.ensure_directories()
