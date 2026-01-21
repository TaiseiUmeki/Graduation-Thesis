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
        output_dir: Optional[Path] = None,
        concept: Optional[str] = None
    ) -> GeneratedImage:
        """
        特徴ベクトルから画像を生成
        
        Args:
            base_image_path: 元画像のパス
            vector: 特徴ベクトル
            constraints: 制約リスト
            output_dir: 出力ディレクトリ
            concept: 物体のコンセプト（例: "Tank"）
        
        Returns:
            生成画像情報
        """
        output_dir = output_dir or Config.IMAGES_DIR
        
        # ベクトルからプロンプトを生成
        prompt = self._build_image_prompt(vector, constraints, concept)
        
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
    
    def generate_synthesis(
        self,
        base_image_path: str,
        mask_path: str,
        instruction: str,
        output_dir: Optional[Path] = None
    ) -> str:
        """
        粘土素材の物理的結合（Inpainting）
        
        Args:
            base_image_path: ベース画像のパス
            mask_path: マスク画像のパス
            instruction: 結合指示（例: "右のパーツを砲台として結合して"）
            output_dir: 出力ディレクトリ
        
        Returns:
            生成された画像のパス
        """
        output_dir = output_dir or Config.IMAGES_DIR
        
        # Inpainting用プロンプト：デザイン変更はせず、粘土として物理的に結合することだけを指示
        prompt = f"""Combine these clay objects physically as instructed below. 
Keep the material as RAW CLAY throughout. Do not change the design or geometry - only join them together.

Instruction: {instruction}

Output: A seamless clay sculpture where the parts are naturally merged."""
        
        print(f"\n合成プロンプト:\n{prompt}\n")
        
        # OpenAI Images APIのinpainting（edit）を使用
        result = self.client.inpaint_image(base_image_path, mask_path, prompt)
        
        if result.get("b64_json"):
            image_path = self.image_utils.save_base64_image(
                result["b64_json"],
                output_dir,
                prefix="synthesis"
            )
        else:
            image_path = self.image_utils.download_image_from_url(
                result["url"],
                output_dir,
                prefix="synthesis"
            )
        
        return image_path

    def generate_part_from_vector(
        self,
        base_image_path: str,
        mask_path: str,
        partial_vector: AttributeVector,
        concept: Optional[str] = None,
        target_part_name: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> str:
        """
        部分編集（インペインティング）用: 部分ベクトルとマスクから編集画像を生成
        Args:
            base_image_path: ベース画像のパス
            mask_path: 透過マスクのパス（透明=編集対象）
            partial_vector: 部分編集に用いる属性ベクトル（関連属性のみ）
            concept: 全体のモチーフ（例: "tank"）
            target_part_name: 編集対象の部位名（例: "turret"）
            output_dir: 出力ディレクトリ
        Returns:
            生成された画像のパス
        """
        output_dir = output_dir or Config.IMAGES_DIR

        # 部分編集プロンプトの構築
        part_label = target_part_name or "part"
        subject = f"A clay {concept}" if concept else "A clay object"

        # 部分属性の列挙（正負を区別）
        attr_lines = []
        top_attrs = self.attr_space.get_top_attributes(partial_vector, top_k=20, threshold=0.0)
        for attr_key, weight in top_attrs:
            name = self.attr_space.get_attribute_name(attr_key)
            if not name:
                continue
            if weight >= 0:
                attr_lines.append(f"{name}({weight:.2f})")
            else:
                attr_lines.append(f"avoid {name}({weight:.2f})")

        attr_text = ", ".join(attr_lines) if attr_lines else "(no attributes specified)"

        prompt = (
            f"Edit only the masked region to adjust the {part_label} of {subject}. "
            f"Keep the entire object fully inside the frame; do not crop. "
            f"Keep MATERIAL as RAW CLAY; do not change other parts. "
            f"Modify only GEOMETRY and form in the masked area using: {attr_text}. "
            f"Output: coherent clay sculpture with the edited {part_label} seamlessly integrated."
        )

        print(f"\n部分編集プロンプト:\n{prompt}\n")

        result = self.client.inpaint_image(base_image_path, mask_path, prompt)

        if result.get("b64_json"):
            image_path = self.image_utils.save_base64_image(
                result["b64_json"],
                output_dir,
                prefix="partial"
            )
        else:
            image_path = self.image_utils.download_image_from_url(
                result["url"],
                output_dir,
                prefix="partial"
            )

        return image_path
    
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
        constraints: Optional[List[Constraint]] = None,
        concept: Optional[str] = None
    ) -> str:
        """画像生成用のプロンプトを構築"""
        prompt_parts = []
        
        # Subject: コンセプトがあれば明示
        if concept:
            prompt_parts.append(f"A 3D render of a {concept} made of CLAY.")
        else:
            prompt_parts.append("A 3D render of an object made of CLAY.")
        # フレーミング: 画角内に収める
        prompt_parts.append("Keep the entire clay object fully inside the frame with no cropping or cut-off edges; center it with a small margin around the subject.")
        
        # 制約: 材質は粘土のまま、形状だけを変更
        prompt_parts.append("【最重要】Keep the material as raw clay. Only modify the GEOMETRY (shape, edges, contours) based on the following attributes.")
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
            
            # グループごとに記述（ATTR_SPACE.groupsの順序を保持）
            attr_descriptions = []
            
            # グループの順序に従って処理
            for group_name in self.attr_space.groups.keys():
                if group_name not in grouped_attrs:
                    continue
                
                # 属性を整形
                attrs_formatted = []
                for name, weight in grouped_attrs[group_name][:Config.MAX_ATTRS_PER_GROUP]:
                    if weight >= 0:
                        attrs_formatted.append(f"{name}({weight:.2f})")
                    else:
                        attrs_formatted.append(f"（{name}を避ける{weight:.2f}）")
                
                if attrs_formatted:
                    attr_descriptions.append(f"{group_name}: {', '.join(attrs_formatted)}")
            
            if attr_descriptions:
                prompt_parts.append("、".join(attr_descriptions))
        
        # 制約の追加
        if constraints:
            active_constraints = [c for c in constraints if c.is_active]
            if active_constraints:
                constraint_texts = []
                for c in active_constraints:
                    if c.description:
                        constraint_texts.append(c.description)
                
                if constraint_texts:
                    prompt_parts.append(f"制約: {', '.join(constraint_texts)}")
        
        # プロンプトを結合
        full_prompt = "。".join(prompt_parts) + "。粘土のテクスチャを保持した、高品質で詳細な3Dレンダリング。"
        
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
