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
    
    # 入力・マスク関連ディレクトリ
    INPUT_IMAGES_DIR = DATA_DIR / "input_images"  # UIからアップロードされた画像
    MASKS_DIR = DATA_DIR / "masks"  # マスク画像全般
    MASKS_PARTIAL_A_DIR = MASKS_DIR / "partialA"  # Phase A部分編集
    MASKS_PARTIAL_D_DIR = MASKS_DIR / "partialD"  # Phase D部分編集
    MASKS_SYNTHESIS_DIR = MASKS_DIR / "synthesis"  # 合成モード
    
    # OpenAI API設定
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # モデル設定
    GPT_MODEL = "gpt-4o"  # クエリ解釈・特徴ベクトル生成用
    VISION_MODEL = "gpt-4o"  # 画像理解用
    IMAGE_MODEL = "gpt-image-1"  # 画像生成用
    
    # パラメータ設定
    TEMPERATURE = 0.7  # 生成の多様性（0.0-2.0）
    MAX_TOKENS = 2000  # 最大トークン数
    NUM_INTERPRETATIONS = 3  # 解釈案の数
    MAX_ATTRS_PER_GROUP = 5  # プロンプト内の属性グループごとの最大属性数
    
    # 画像生成設定
    IMAGE_SIZE = "1024x1024"  # 生成画像サイズ
    IMAGE_QUALITY = "auto"  # low, medium, high, auto
    IMAGE_RESPONSE_FORMAT = "b64_json"  # url or b64_json
    
    @classmethod
    def ensure_directories(cls):
        """必要なディレクトリを作成"""
        dir_list = [
            cls.DATA_DIR,
            cls.SESSIONS_DIR,
            cls.IMAGES_DIR,
            cls.OUTPUTS_DIR,
            cls.INPUT_IMAGES_DIR,
            cls.MASKS_DIR,
            cls.MASKS_PARTIAL_A_DIR,
            cls.MASKS_PARTIAL_D_DIR,
            cls.MASKS_SYNTHESIS_DIR,
        ]
        for dir_path in dir_list:
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
