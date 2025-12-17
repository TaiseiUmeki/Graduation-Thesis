"""
OpenAI APIクライアントモジュール
GPT-4, GPT-4 Vision, DALL-E 3のAPIラッパー
"""
from typing import List, Optional, Dict, Any
import base64
from pathlib import Path
import json

from openai import OpenAI

from config import Config


class OpenAIClient:
    """OpenAI APIのクライアントクラス"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API keyが設定されていません")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        チャット補完APIを呼び出し
        
        Args:
            messages: メッセージリスト
            model: 使用するモデル
            temperature: 生成の多様性
            max_tokens: 最大トークン数
            response_format: レスポンス形式（JSONモード用）
        
        Returns:
            生成されたテキスト
        """
        model = model or Config.GPT_MODEL
        temperature = temperature if temperature is not None else Config.TEMPERATURE
        max_tokens = max_tokens or Config.MAX_TOKENS
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def analyze_image_with_text(
        self,
        image_path: str,
        prompt: str,
        model: Optional[str] = None
    ) -> str:
        """
        画像とテキストを組み合わせて分析
        
        Args:
            image_path: 画像ファイルのパス
            prompt: 質問・指示文
            model: 使用するモデル（デフォルト: gpt-4o）
        
        Returns:
            分析結果のテキスト
        """
        model = model or Config.VISION_MODEL
        
        # 画像をbase64エンコード
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 画像の拡張子から形式を判定
        image_format = Path(image_path).suffix.lower().replace('.', '')
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        return self.chat_completion(messages, model=model)
    
    def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        DALL-E 3で画像を生成
        
        Args:
            prompt: 画像生成プロンプト
            size: 画像サイズ（"1024x1024", "1792x1024", "1024x1792"）
            quality: 品質（"standard" or "hd"）
            model: モデル名（デフォルト: dall-e-3）
        
        Returns:
            {"url": 画像URL, "revised_prompt": 改訂されたプロンプト}
        """
        model = model or Config.IMAGE_MODEL
        size = size or Config.IMAGE_SIZE
        quality = quality or Config.IMAGE_QUALITY
        
        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1
        )
        
        return {
            "url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt or prompt
        }
    
    def edit_image(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        size: Optional[str] = None
    ) -> str:
        """
        DALL-E 2で画像を編集（インペインティング）
        注: DALL-E 3は編集をサポートしていないため、DALL-E 2を使用
        
        Args:
            image_path: 元画像のパス
            mask_path: マスク画像のパス
            prompt: 編集内容の指示
            size: 画像サイズ
        
        Returns:
            編集された画像のURL
        """
        size = size or "1024x1024"
        
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            response = self.client.images.edit(
                model="dall-e-2",
                image=image_file,
                mask=mask_file,
                prompt=prompt,
                size=size,
                n=1
            )
        
        return response.data[0].url
    
    def generate_with_json_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict:
        """
        JSON形式でレスポンスを取得
        
        Args:
            prompt: プロンプト
            system_prompt: システムプロンプト
            model: モデル名
        
        Returns:
            パースされたJSON辞書
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response_text = self.chat_completion(
            messages,
            model=model,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response_text)
    
    def create_structured_prompt(
        self,
        instruction: str,
        context: Optional[str] = None,
        examples: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        構造化されたプロンプトを作成
        
        Args:
            instruction: 指示
            context: コンテキスト
            examples: 例
            constraints: 制約
        
        Returns:
            構造化されたプロンプト文字列
        """
        parts = []
        
        if context:
            parts.append(f"# コンテキスト\n{context}\n")
        
        parts.append(f"# 指示\n{instruction}\n")
        
        if examples:
            parts.append("# 例")
            for i, ex in enumerate(examples, 1):
                parts.append(f"{i}. {ex}")
            parts.append("")
        
        if constraints:
            parts.append("# 制約")
            for i, con in enumerate(constraints, 1):
                parts.append(f"{i}. {con}")
            parts.append("")
        
        return "\n".join(parts)
