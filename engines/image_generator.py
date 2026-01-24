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
        concept: Optional[str] = None,
        motif: Optional[str] = None
    ) -> GeneratedImage:
        """
        特徴ベクトルから画像を生成
        
        Args:
            base_image_path: 元画像のパス
            vector: 特徴ベクトル
            constraints: 制約リスト
            output_dir: 出力ディレクトリ
            concept: 物体のコンセプト（例: "Tank"）
            motif: モチーフ（固有名詞、例: "Tulip"）
        
        Returns:
            生成画像情報
        """
        output_dir = output_dir or Config.IMAGES_DIR
        
        # ベクトルからプロンプトを生成
        prompt = self._build_image_prompt(vector, constraints, concept, motif)
        
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
        part_name: str,          # 変更: 左側のパーツ名
        body_name: str,          # 変更: 右側の本体名
        synthesis_method: Optional[str] = None,
        user_intent: Optional[str] = None, # 追加: ユーザーの意図（任意）
        output_dir: Optional[Path] = None
    ) -> str:
        """
        粘土素材の物理的結合（Inpainting）
        構造化されたプロンプトを使用
        """
        output_dir = output_dir or Config.IMAGES_DIR
        
        # 構造化されたプロンプト構築
        prompt_parts = [
            # 1. 役割定義と素材制約
            "You are a professional sculptor combining two clay objects.",
            "Keep the material as RAW CLAY throughout. Do not turn them into real objects.",
            "Do not change the overall geometry outside the masked region.",
            
            # 2. 空間・対象の構造化定義
            f"The object on the LEFT side of the image is the '{part_name}' (part).",
            f"The object on the RIGHT side of the image is the '{body_name}' (main body).",
            
            # 3. タスク指示
            f"Task: Physically join the '{part_name}' onto the '{body_name}' within the masked region.",
        ]
        
        # 4. 接合方法（解釈）の反映
        if synthesis_method:
            prompt_parts.append(f"Joint Style: {synthesis_method}")
            
        # 5. ユーザーの追加意図（あれば）
        if user_intent:
            prompt_parts.append(f"User instruction: {user_intent}")
        
        prompt_parts.append("Output: A seamless clay sculpture where the parts are joined naturally.")
        
        prompt = " ".join(prompt_parts)
        
        print(f"\n合成プロンプト:\n{prompt}\n")
        
        # OpenAI Images APIのinpaintingを実行
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
        partial_motif: Optional[str] = None,
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
            partial_motif: 部分編集のモチーフ（解釈フェーズで抽出された固有名詞）
            output_dir: 出力ディレクトリ
        Returns:
            生成された画像のパス
        """
        output_dir = output_dir or Config.IMAGES_DIR

        # 部分編集プロンプトの構築
        part_label = target_part_name or "part"
        subject = f"A clay {concept}" if concept else "A clay object"

        # 部分属性の列挙
        attr_lines = []
        top_attrs = self.attr_space.get_top_attributes(partial_vector, top_k=20, threshold=0.0)
        for attr_key, weight in top_attrs:
            name = self.attr_space.get_attribute_name(attr_key)
            if not name:
                continue
            attr_lines.append(f"{name}({weight:.2f})")

        attr_text = ", ".join(attr_lines) if attr_lines else "(no attributes specified)"

        prompt_parts = [
            f"Edit only the masked region to adjust the {part_label} of {subject}.",
            "Keep the entire object fully inside the frame; do not crop.",
            "Keep MATERIAL as RAW CLAY; do not change other parts.",
            f"Modify only GEOMETRY and form in the masked area using: {attr_text}.",
        ]

        # 解釈フェーズで抽出されたモチーフを使用
        if partial_motif:
            prompt_parts.append(f"Incorporate the characteristic shape and style of a {partial_motif} into the edited part.")

        prompt_parts.append("Output: coherent clay sculpture with the edited part seamlessly integrated.")

        prompt = " ".join(prompt_parts)

        print(f"\n部分編集プロンプト:\n{prompt}\n")

        # マスクとベース画像のサイズを確認・調整
        from PIL import Image
        base_image = Image.open(base_image_path)
        mask_image = Image.open(mask_path)
        
        # サイズが異なる場合はマスクをベース画像のサイズにリサイズ
        if base_image.size != mask_image.size:
            print(f"サイズ不一致を検出: ベース画像 {base_image.size} vs マスク {mask_image.size}")
            print("マスクをベース画像のサイズにリサイズしています...")
            
            # マスクをベース画像のサイズにリサイズ
            resized_mask = mask_image.resize(base_image.size, Image.Resampling.NEAREST)
            
            # 一時的なマスクファイルとして保存
            import tempfile
            import os
            temp_mask_fd, temp_mask_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_mask_fd)
            resized_mask.save(temp_mask_path)
            
            try:
                result = self.client.inpaint_image(base_image_path, temp_mask_path, prompt)
            finally:
                # 一時ファイルを削除
                os.unlink(temp_mask_path)
        else:
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
        concept: Optional[str] = None,
        motif: Optional[str] = None
    ) -> str:
        """画像生成用のプロンプトを構築"""
        prompt_parts = []
        
        # Subject: コンセプトがあれば明示
        if concept:
            prompt_parts.append(f"A 3D render of a {concept} made of CLAY.")
        else:
            prompt_parts.append("A 3D render of an object made of CLAY.")
        
        # Motif Injection: モチーフが指定されている場合、形状の比喩として組み込む
        if motif:
            prompt_parts.append(f"It is designed with the distinct motif of a {motif}.")
            prompt_parts.append(f"Incorporate the characteristic shape and silhouette of a {motif} into the design.")
        
        # フレーミング: 画角内に収める
        prompt_parts.append("Keep the entire clay object fully inside the frame with no cropping or cut-off edges; center it with a small margin around the subject.")
        
        # 制約: 材質は粘土のまま、形状だけを変更
        prompt_parts.append("【最重要】Keep the material as raw clay. Only modify the GEOMETRY (shape, edges, contours) based on the following attributes.")
        prompt_parts.append("（注：各属性の値は0.0～1.0の範囲です。値が大きいほどその特徴を強調します）")
        
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
                    attrs_formatted.append(f"{name}({weight:.2f})")
                
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
