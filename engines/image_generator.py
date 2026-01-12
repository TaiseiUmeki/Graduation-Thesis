"""
画像生成エンジン（フェーズD）
特徴ベクトルから画像を生成し、批評ループを管理
"""
from typing import List, Optional, Tuple
from pathlib import Path

from utils.openai_client import OpenAIClient
from utils.image_utils import ImageUtils
from models.attribute_space import AttributeVector, ATTR_SPACE
from models.session import GeneratedImage
from models.constraints import Constraint
from config import Config


class ImageGenerator:
    """画像生成エンジン"""
    
    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client or OpenAIClient()
        self.attr_space = ATTR_SPACE
        self.image_utils = ImageUtils()
    
    def generate_from_vector(
        self,
        base_image_path: str,
        vector: AttributeVector,
        constraints: Optional[List[Constraint]] = None,
        output_dir: Optional[Path] = None
    ) -> GeneratedImage:
        """
        特徴ベクトルから画像を生成
        
        Args:
            base_image_path: 元画像のパス
            vector: 特徴ベクトル
            constraints: 制約リスト
            output_dir: 出力ディレクトリ
        
        Returns:
            生成画像情報
        """
        output_dir = output_dir or Config.IMAGES_DIR
        
        # ベクトルからプロンプトを生成
        prompt = self._build_image_prompt(vector, constraints)
        
        print(f"\n画像生成プロンプト:\n{prompt}\n")
        
        # 参照画像 + プロンプトで画像生成
        result = self.client.generate_image_from_image(base_image_path, prompt)
        
        if result.get("b64_json"):
            image_path = self.image_utils.save_base64_image(
                result["b64_json"],
                output_dir,
                prefix="generated"
            )
        else:
            image_path = self.image_utils.download_image_from_url(
                result["url"],
                output_dir,
                prefix="generated"
            )
        
        # GeneratedImageオブジェクトを作成
        generated_image = GeneratedImage(
            image_path=image_path,
            prompt=prompt,
            vector=vector,
            constraints=constraints or []
        )
        
        return generated_image
    
    def suggest_constraint_attributes(
        self,
        vector: AttributeVector,
        num_suggestions: int = 5
    ) -> List[Tuple[str, str, float, str]]:
        """
        ユーザーに提案する制約候補の属性を選択
        
        Args:
            vector: 現在の特徴ベクトル
            num_suggestions: 提案する属性の数
        
        Returns:
            (属性キー, 属性名, 現在の値, グループ名)のリスト
        """
        # ベクトルから上位の属性を取得
        top_attrs = self.attr_space.get_top_attributes(vector, top_k=20)
        
        # 多様なグループから選択
        suggestions = []
        used_groups = set()
        
        for attr_key, weight in top_attrs:
            group_name = attr_key.split(':')[0]
            
            # 既に同じグループから選んでいない、かつ有意な重みがある場合
            if group_name not in used_groups and weight > 0.3:
                attr_name = self.attr_space.get_attribute_name(attr_key)
                if attr_name:
                    suggestions.append((attr_key, attr_name, weight, group_name))
                    used_groups.add(group_name)
                
                if len(suggestions) >= num_suggestions:
                    break
        
        return suggestions
    
    def analyze_image_with_gpt(
        self,
        image_path: str,
        analysis_prompt: Optional[str] = None
    ) -> str:
        """
        生成された画像をGPT-4 Visionで分析
        
        Args:
            image_path: 画像パス
            analysis_prompt: 分析用のプロンプト
        
        Returns:
            分析結果のテキスト
        """
        if analysis_prompt is None:
            analysis_prompt = """この画像について以下の観点で分析してください：
1. 主な材質や質感
2. 形状の特徴
3. 色彩や配色
4. 全体的な印象やスタイル
5. 改善提案があれば

簡潔に箇条書きで答えてください。"""
        
        return self.client.analyze_image_with_text(image_path, analysis_prompt)
    
    def compare_images(
        self,
        image1_path: str,
        image2_path: str,
        comparison_focus: Optional[str] = None
    ) -> str:
        """
        2つの画像を比較
        
        Args:
            image1_path: 画像1のパス
            image2_path: 画像2のパス
            comparison_focus: 比較の焦点
        
        Returns:
            比較結果のテキスト
        """
        focus_text = comparison_focus or "全体的な違い"
        
        prompt = f"""2つの画像を比較して、{focus_text}について説明してください。
特に変化した点や改善された点を具体的に指摘してください。"""
        
        # 両方の画像を読み込んでbase64エンコード
        import base64
        
        with open(image1_path, "rb") as f1:
            image1_data = base64.b64encode(f1.read()).decode('utf-8')
        with open(image2_path, "rb") as f2:
            image2_data = base64.b64encode(f2.read()).decode('utf-8')
        
        # 画像形式を判定
        ext1 = Path(image1_path).suffix.lower().replace('.', '')
        ext2 = Path(image2_path).suffix.lower().replace('.', '')
        if ext1 == 'jpg': ext1 = 'jpeg'
        if ext2 == 'jpg': ext2 = 'jpeg'
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext1};base64,{image1_data}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext2};base64,{image2_data}"}
                    }
                ]
            }
        ]
        
        return self.client.chat_completion(messages, model=Config.VISION_MODEL)
    
    def _build_image_prompt(
        self,
        vector: AttributeVector,
        constraints: Optional[List[Constraint]] = None
    ) -> str:
        """画像生成用のプロンプトを構築"""
        prompt_parts = []
        prompt_parts.append("参照画像をベースに、以下の特徴を反映して新しい画像を生成してください。")
        prompt_parts.append("（注：各属性の値は-1.0～1.0の範囲です。正の値は特徴を強調し、負の値はその特徴を積極的に削ぎ落とします）")
        
        # 特徴ベクトルから主要な属性を抽出（負値も含める）
        top_attributes = self.attr_space.get_top_attributes(vector, top_k=20, threshold=0.0)
        
        if top_attributes:
            # 属性をグループごとに整理（重み値付き、負値対応）
            grouped_attrs = {}
            for attr_key, weight in top_attributes:
                group_name = attr_key.split(':')[0]
                attr_name = self.attr_space.get_attribute_name(attr_key)
                
                if attr_name:
                    if group_name not in grouped_attrs:
                        grouped_attrs[group_name] = []
                    grouped_attrs[group_name].append((attr_name, weight))
            
            # グループごとに記述
            attr_descriptions = []
            
            # 材質
            if "material" in grouped_attrs:
                materials = []
                for name, weight in grouped_attrs["material"][:3]:
                    if weight >= 0:
                        materials.append(f"{name}({weight:.2f})")
                    else:
                        materials.append(f"（{name}を避ける{weight:.2f}）")
                if materials:
                    attr_descriptions.append(f"材質: {', '.join(materials)}")
            
            # 仕上げ
            if "finish" in grouped_attrs:
                finishes = []
                for name, weight in grouped_attrs["finish"][:3]:
                    if weight >= 0:
                        finishes.append(f"{name}({weight:.2f})")
                    else:
                        finishes.append(f"（{name}を避ける{weight:.2f}）")
                if finishes:
                    attr_descriptions.append(f"仕上げ: {', '.join(finishes)}")
            
            # 形状
            if "shape" in grouped_attrs:
                shapes = []
                for name, weight in grouped_attrs["shape"][:3]:
                    if weight >= 0:
                        shapes.append(f"{name}({weight:.2f})")
                    else:
                        shapes.append(f"（{name}を避ける{weight:.2f}）")
                if shapes:
                    attr_descriptions.append(f"形状: {', '.join(shapes)}")
            
            # 色
            if "color" in grouped_attrs:
                colors = []
                for name, weight in grouped_attrs["color"][:3]:
                    if weight >= 0:
                        colors.append(f"{name}({weight:.2f})")
                    else:
                        colors.append(f"（{name}を避ける{weight:.2f}）")
                if colors:
                    attr_descriptions.append(f"色: {', '.join(colors)}")
            
            # 機能的な見た目
            if "function_visual" in grouped_attrs:
                functions = []
                for name, weight in grouped_attrs["function_visual"][:2]:
                    if weight >= 0:
                        functions.append(f"{name}({weight:.2f})")
                    else:
                        functions.append(f"（{name}を避ける{weight:.2f}）")
                if functions:
                    attr_descriptions.append(f"機能的要素: {', '.join(functions)}")
            
            # パターン・模様
            if "pattern" in grouped_attrs:
                patterns = []
                for name, weight in grouped_attrs["pattern"][:2]:
                    if weight >= 0:
                        patterns.append(f"{name}({weight:.2f})")
                    else:
                        patterns.append(f"（{name}を避ける{weight:.2f}）")
                if patterns:
                    attr_descriptions.append(f"パターン: {', '.join(patterns)}")
            
            if attr_descriptions:
                prompt_parts.append("、".join(attr_descriptions))
        
        # 制約の追加
        if constraints:
            active_constraints = [c for c in constraints if c.is_active]
            if active_constraints:
                constraint_texts = []
                for c in active_constraints[-3:]:  # 直近3つまで
                    if c.description:
                        constraint_texts.append(c.description)
                
                if constraint_texts:
                    prompt_parts.append(f"制約: {', '.join(constraint_texts)}")
        
        # プロンプトを結合
        full_prompt = "。".join(prompt_parts) + "。高品質で詳細な3Dレンダリング。"
        
        return full_prompt
    
    def extract_description_from_image(
        self,
        image_path: str
    ) -> str:
        """
        画像から説明を抽出（粘土の中間生成物の説明用）
        
        Args:
            image_path: 画像パス
        
        Returns:
            画像の説明文
        """
        prompt = """この画像に写っている物体について、以下の情報を簡潔に説明してください：
1. 基本的な形状
2. サイズ感
3. 現在の材質（粘土など）
4. 特徴的な部分

「この物体は〜」という形式で、1-2文で説明してください。"""
        
        return self.client.analyze_image_with_text(image_path, prompt)
