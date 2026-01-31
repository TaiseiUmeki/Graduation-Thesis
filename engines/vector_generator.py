"""
特徴ベクトル生成エンジン（フェーズC）
解釈案から特徴ベクトルを生成し、関連属性を推薦
"""
from typing import List, Dict, Optional, Tuple
import json
import numpy as np

from utils.openai_client import OpenAIClient
from models.attribute_space import AttributeVector, ATTR_SPACE
from models.session import Interpretation
from models.constraints import Constraint, ConstraintType
from config import Config


class VectorGenerator:
    """特徴ベクトル生成エンジン"""
    
    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client or OpenAIClient()
        self.attr_space = ATTR_SPACE
    
    def generate_vector_from_interpretation(
        self,
        interpretation: Interpretation,
        constraints: Optional[List[Constraint]] = None
    ) -> AttributeVector:
        """
        解釈案から特徴ベクトルを生成
        
        Args:
            interpretation: 解釈案
            constraints: 既存の制約
        
        Returns:
            特徴ベクトル
        """
        # プロンプトの構築
        prompt = self._build_vector_generation_prompt(interpretation, constraints)
        
        # GPTで属性の重みを生成
        response = self.client.generate_with_json_response(
            prompt,
            system_prompt=self._get_vector_system_prompt()
        )
        
        # レスポンスから特徴ベクトルを構築
        vector = self._parse_vector_response(response)
        
        return vector

    def generate_from_text(
        self,
        text: str,
        concept: Optional[str] = None,
        max_attrs: int = 8
    ) -> AttributeVector:
        """
        解釈を介さず自由テキストから部分編集向けのベクトルを生成（ゼロベクトル起点）
        - 既存の属性カタログに限定し、関連する少数属性のみを選ぶ
        Args:
            text: ユーザー入力テキスト（部位名や意図を含めても良い）
            concept: モチーフ（文脈付与用）
            max_attrs: 最大属性数
        Returns:
            部分編集用のAttributeVector
        """
        # まずテキストから関連属性キーを抽出
        related_keys = self.map_text_to_attributes(text, max_results=max_attrs, return_expanded=False) or []

        # 抽出された属性のみに対して重みを付与するJSON出力を促す
        catalog_lines = []
        for key in related_keys:
            name = self.attr_space.get_attribute_name(key) or key
            catalog_lines.append(f"  {key} = {name}")
        catalog_text = "\n".join(catalog_lines) or "  (none)"

        system_prompt = (
            "あなたは部分編集用の属性重みを決めるアシスタントです。"
            "与えられた候補属性のみを使い、0.0～1.0の範囲で重みを設定してください。値が大きいほど強調されます。"
            "重みはテキスト意図に基づき、必要最小限のみ非ゼロにしてください。"
        )

        user_prompt = f"""
対象コンセプト: {concept or '(未指定)'}
ユーザー意図テキスト: "{text}"

候補属性（既存カタログから抽出済み）:
{catalog_text}

JSON形式で返してください:
{{
  "attributes": {{
    "group:key": 0.6,
    ... (候補属性のみに限定)
  }},
  "reasoning": "なぜその属性を選んだかの短い説明"
}}
"""

        response = self.client.generate_with_json_response(
            user_prompt,
            system_prompt=system_prompt
        )

        vector = self._parse_vector_response(response)
        return vector

    def generate_global_from_image(
        self,
        image_path: str,
        concept: Optional[str] = None
    ) -> AttributeVector:
        """
        画像分析からグローバルベクトルを生成（Phase AのRoot向け）
        - 画像をVisionで分析し、その説明テキストから属性重みを作成
        """
        analysis_prompt = """この画像の粘土物体について、形状・エッジ・輪郭・材質に関する要点を簡潔に記述してください。
（例: 角張り/丸み、直線/曲線、対称性、粗い/滑らか など）"""
        description = self.client.analyze_image_with_text(image_path, analysis_prompt)

        # 説明テキストから属性ベクトル（重要属性のみ）
        return self.generate_from_text(text=description, concept=concept, max_attrs=10)
    
    def find_related_attributes(
        self,
        interpretation: Interpretation,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        解釈案とクエリに関連性の高い属性を検索
        
        Args:
            interpretation: 解釈案
            query: 元のクエリ
            top_k: 返す属性の数
        
        Returns:
            (属性キー, 属性名, 関連度スコア)のリスト
        """
        prompt = self._build_attribute_search_prompt(interpretation, query, top_k)
        
        response = self.client.generate_with_json_response(
            prompt,
            system_prompt=self._get_attribute_search_system_prompt()
        )
        
        # レスポンスをパース
        related_attrs = self._parse_related_attributes(response)
        
        return related_attrs
    
    def update_vector_with_constraint(
        self,
        current_vector: AttributeVector,
        constraint: Constraint,
        interpretation: Optional[Interpretation]
    ) -> AttributeVector:
        """
        制約を考慮して特徴ベクトルを更新
        
        Args:
            current_vector: 現在の特徴ベクトル
            constraint: 新しい制約
            interpretation: 解釈案（None可）
        
        Returns:
            更新された特徴ベクトル
        """
        prompt = self._build_constraint_update_prompt(
            current_vector,
            constraint,
            interpretation
        )
        
        response = self.client.generate_with_json_response(
            prompt,
            system_prompt=self._get_vector_system_prompt()
        )
        
        updated_vector = self._parse_vector_response(response)
        
        return updated_vector

    def update_vector_with_constraints(
        self,
        base_vector: AttributeVector,
        constraints: List[Constraint]
    ) -> AttributeVector:
        """
        複数制約を決定論的に適用して特徴ベクトルを更新（LLM不使用）

        Phase D（Global）のスライダー操作に対応するため、ConstraintType に従って
        base_vector の重みを更新する。
        """
        if not constraints:
            return base_vector

        updated_weights = dict(base_vector.weights)

        for c in constraints:
            if not getattr(c, "is_active", True):
                continue

            attr_key = c.attribute
            if not self.attr_space.has_attribute(attr_key):
                continue

            current_val = float(updated_weights.get(attr_key, 0.0))

            ctype = getattr(c.constraint_type, "value", c.constraint_type)

            if ctype == ConstraintType.EQUAL.value and c.value is not None:
                updated_weights[attr_key] = float(np.clip(float(c.value), 0.0, 1.0))
            elif ctype == ConstraintType.GREATER_THAN.value and c.value is not None:
                updated_weights[attr_key] = float(np.clip(max(current_val, float(c.value)), 0.0, 1.0))
            elif ctype == ConstraintType.LESS_THAN.value and c.value is not None:
                updated_weights[attr_key] = float(np.clip(min(current_val, float(c.value)), 0.0, 1.0))
            elif (
                ctype == ConstraintType.RANGE.value
                and c.min_value is not None
                and c.max_value is not None
            ):
                updated_weights[attr_key] = float(
                    np.clip(current_val, float(c.min_value), float(c.max_value))
                )

        return AttributeVector(weights=updated_weights)
    
    def _get_vector_system_prompt(self) -> str:
        """ベクトル生成用のシステムプロンプト"""
        return """あなたは創作物の特徴を属性ベクトルで表現する専門家です。
与えられた解釈や説明から、適切な属性とその重み（0.0～1.0：値が大きいほど強調）を決定してください。

スコアリングの基準:
- 1.0: 非常に強く該当する
- 0.7: 強く該当する
- 0.5: 中程度に該当する
- 0.3: 弱く該当する
- 0.0: その属性は全く該当しない

制約がある場合は、それを必ず満たすように重みを設定してください。"""
    
    def _get_attribute_search_system_prompt(self) -> str:
        """属性検索用のシステムプロンプト"""
        return """あなたは創作物の特徴を分析する専門家です。
与えられた解釈やクエリに関連性の高い属性を、属性リストから選んでください。
関連度スコアは0.0～1.0の範囲で、値が大きいほど関連度が高いことを示します。"""
    
    def _build_vector_generation_prompt(
        self,
        interpretation: Interpretation,
        constraints: Optional[List[Constraint]] = None
    ) -> str:
        """ベクトル生成用のプロンプトを構築"""
        # 属性グループの情報を整形
        attr_info = self._format_attribute_groups()
        
        prompt_parts = [
            "以下の解釈に基づいて、特徴ベクトルを生成してください。",
            f"\n解釈: {interpretation.text}",
            f"根拠: {interpretation.reasoning}",
        ]
        
        if constraints:
            prompt_parts.append("\n制約条件:")
            for c in constraints:
                if c.is_active:
                    prompt_parts.append(f"- {c.to_natural_language()}")
        
        prompt_parts.append(f"\n利用可能な属性:\n{attr_info}")
        
        prompt_parts.append("""
以下のJSON形式で、関連する属性とその重み（0.0～1.0）を返してください。
重要な属性のみを含め、重み0.3未満の属性は省略してください。

{
  "attributes": {
        "form:cubic": 0.8,
        "edge:filleted": 0.6,
    ...
  },
  "reasoning": "各属性を選んだ理由の簡単な説明"
}
""")
        
        return "\n".join(prompt_parts)
    
    def _build_attribute_search_prompt(
        self,
        interpretation: Interpretation,
        query: str,
        top_k: int
    ) -> str:
        """属性検索用のプロンプトを構築"""
        attr_info = self._format_attribute_groups()
        
        prompt = f"""以下のクエリと解釈に最も関連性の高い属性を{top_k}個選んでください。

クエリ: {query}
解釈: {interpretation.text}

利用可能な属性:
{attr_info}

以下のJSON形式で返してください：
{{
  "related_attributes": [
    {{
      "attribute_key": "form:cubic",
      "attribute_name": "立方体・箱型",
      "relevance_score": 0.9,
      "reason": "選んだ理由"
    }},
    ...
  ]
}}
"""
        return prompt
    
    def _build_constraint_update_prompt(
        self,
        current_vector: AttributeVector,
        constraint: Constraint,
        interpretation: Optional[Interpretation]
    ) -> str:
        """制約を考慮したベクトル更新用のプロンプト"""
        # 現在の主要な属性を取得
        top_attrs = self.attr_space.get_top_attributes(current_vector, top_k=10)
        current_attrs_text = "\n".join([
            f"- {attr}: {weight:.2f}" for attr, weight in top_attrs
        ])
        
        # 解釈テキストを取得（None の場合は空文字列）
        interpretation_text = interpretation.text if interpretation else "（解釈なし）"
        
        prompt = f"""現在の特徴ベクトルに新しい制約を適用して更新してください。

解釈: {interpretation_text}

現在の主要属性:
{current_attrs_text}

新しい制約: {constraint.to_natural_language()}
説明: {constraint.description}

この制約を満たすように、特徴ベクトルの重みを調整してください。
制約に関連する属性だけでなく、バランスを保つために他の属性も調整してください。

以下のJSON形式で返してください：
{{
  "attributes": {{
    "form:cubic": 0.8,
    "edge:sharp_angle": 0.6,
    ...
  }},
  "reasoning": "どのように調整したかの説明"
}}
"""
        return prompt
    
    def _format_attribute_groups(self) -> str:
        """属性グループを読みやすく整形"""
        lines = []
        lines.append("※重要: 以下の属性リストにない属性を作成しないでください")
        for group_name, attrs in self.attr_space.groups.items():
            lines.append(f"\n【{group_name}】")
            for key, name in attrs.items():  # 全件表示に変更
                full_key = f"{group_name}:{key}"
                lines.append(f"  {full_key} = {name}")
        
        return "\n".join(lines)
    
    def _parse_vector_response(self, response: Dict) -> AttributeVector:
        """レスポンスから特徴ベクトルを構築"""
        attributes = response.get("attributes", {})
        
        # 属性キーの検証と正規化
        validated_weights = self._sanitize_weights(attributes)
        
        return AttributeVector(weights=validated_weights)
    
    def _sanitize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """属性重みを正規化し、不明な属性を除去"""
        validated_weights = {}
        invalid_attrs = []
        
        for attr_key, weight in weights.items():
            if self.attr_space.has_attribute(attr_key):
                validated_weights[attr_key] = max(0.0, min(1.0, float(weight)))
            else:
                invalid_attrs.append(attr_key)
        
        if invalid_attrs:
            print(f"警告: 不明な属性を無視しました: {', '.join(invalid_attrs)}")
            print("  → LLMが属性カタログにない属性を生成しています")
        
        return validated_weights
    
    def _parse_related_attributes(
        self,
        response: Dict
    ) -> List[Tuple[str, str, float]]:
        """関連属性のレスポンスをパース"""
        related = []
        for item in response.get("related_attributes", []):
            attr_key = item.get("attribute_key", "")
            attr_name = item.get("attribute_name", "")
            score = float(item.get("relevance_score", 0.0))
            
            if attr_key in self.attr_space.all_attributes:
                related.append((attr_key, attr_name, score))
        
        return related

    def map_text_to_attributes(
        self,
        user_text: str,
        max_results: int = 5,
        return_expanded: bool = True
    ):
        """
        自由テキストを属性空間の既存属性にマッピング（捏造禁止）
        
        Args:
            user_text: ユーザーの自由入力
            max_results: 返却最大件数
            return_expanded: Trueなら拡張版、Falseなら簡易版
        
        Returns:
            拡張版: [{"attribute_key": str, "attribute_name": str}, ...]
            簡易版: ["group:key", ...]
        """
        # 属性カタログ（全件）
        catalog_lines = []
        for group_name, attrs in self.attr_space.groups.items():
            catalog_lines.append(f"[{group_name}]")
            for key, name in attrs.items():
                catalog_lines.append(f"  {group_name}:{key} = {name}")
        catalog_text = "\n".join(catalog_lines)

        system_prompt = (
            "あなたはユーザー入力を属性空間の既存属性にマッピングするアシスタントです。"
            "提供されたカタログ以外の属性を作成してはなりません。"
            "該当が全くない場合は空リストを返してください。"
            "出力はJSONのみで、追加説明は含めないでください。"
        )

        if return_expanded:
            output_format = (
                '{\n  "attributes": [\n'
                '    {"attribute_key": "group:key", "attribute_name": "日本語名"}\n'
                '  ]\n}'
            )
        else:
            output_format = (
                '{\n  "attributes": [\n'
                '    "group:key"\n'
                '  ]\n}'
            )

        user_prompt = f"""
ユーザー入力: "{user_text}"

属性カタログ:
{catalog_text}

制約:
- カタログに存在しない属性を返してはいけません
- 最大 {max_results} 件まで
- 該当が全くない場合は空のリスト
- 出力はJSONのみ

出力形式:
{output_format}
"""

        response = self.client.generate_with_json_response(
            user_prompt,
            system_prompt=system_prompt
        )

        raw_attrs = response.get("attributes", []) or []

        # 正規化と検証
        if return_expanded:
            results = []
            for item in raw_attrs:
                try:
                    key = item.get("attribute_key", "")
                    name = item.get("attribute_name", "")
                except (AttributeError, TypeError):
                    # 文字列だけ来た場合
                    key = str(item) if item else ""
                    name = self.attr_space.get_attribute_name(key) or ""

                if key in self.attr_space.all_attributes:
                    # nameが空なら辞書から補完
                    if not name:
                        name = self.attr_space.get_attribute_name(key) or ""
                    results.append({"attribute_key": key, "attribute_name": name})
                if len(results) >= max_results:
                    break
            return results
        else:
            results_keys = []
            for item in raw_attrs:
                key = item if isinstance(item, str) else (item.get("attribute_key", "") if isinstance(item, dict) else "")
                if key in self.attr_space.all_attributes:
                    results_keys.append(key)
                if len(results_keys) >= max_results:
                    break
            return results_keys

    def suggest_attribute_adjustments_from_text(
        self,
        user_text: str,
        current_vector: AttributeVector,
        max_suggestions: int = 6,
        default_delta: float = 0.25
    ) -> List[Tuple[str, float, str]]:
        """
        ユーザーの自由入力テキストを属性調整候補にマッピング
        戻り値: [(attribute_key, delta, reason), ...]
        """
        # 現在の上位属性（文脈として渡す）
        top_attrs = self.attr_space.get_top_attributes(
            current_vector,
            top_k=12,
            threshold=0.0
        )

        top_attrs_text = "\n".join(
            [f"- {attr_key} ({self.attr_space.get_attribute_name(attr_key)}) = {weight:.2f}" for attr_key, weight in top_attrs]
        ) or "- なし"

        # 属性カタログ（各グループから最大10件）
        catalog_lines = []
        for group_name, attrs in self.attr_space.groups.items():
            catalog_lines.append(f"[{group_name}]")
            for key, name in list(attrs.items())[:10]:
                catalog_lines.append(f"  {group_name}:{key} = {name}")
            if len(attrs) > 10:
                catalog_lines.append(f"  ...他{len(attrs) - 10}件")
        catalog_text = "\n".join(catalog_lines)

        system_prompt = (
            "あなたはプロダクトデザイン用のパラメータ調整アシスタントです。"\
            "属性は 0.0～1.0 の実数で、値が大きいほどその特徴を強調します。"\
            "指定された属性リスト以外は使わないでください。"
        )

        user_prompt = f"""
ユーザー入力: "{user_text}"
現在の主要属性:
{top_attrs_text}

利用可能な属性一覧（抜粋）:
{catalog_text}

指示:
- ユーザー意図に合う属性を最大 {max_suggestions} 件選び、deltaを -0.5～0.5 で提案してください（現在値からの変化量）。
- 典型的な調整幅の初期値は ±{default_delta:.2f} とし、必要に応じて増減してください。
- delta>0 なら強調、delta<0 なら抑制/回避。
- 属性キーは上記リストのものだけを使用。
- JSON形式で返すこと。

出力形式:
{{
  "adjustments": [
    {{"attribute_key": "group:key", "delta": 0.3, "reason": "why"}},
    ... (最大 {max_suggestions} 件)
  ]
}}
"""

        response = self.client.generate_with_json_response(
            user_prompt,
            system_prompt=system_prompt
        )

        adjustments = []
        for item in response.get("adjustments", []):
            attr_key = item.get("attribute_key")
            delta = float(item.get("delta", 0.0))
            reason = item.get("reason", "")

            if attr_key in self.attr_space.all_attributes:
                # クリップして登録
                delta = float(np.clip(delta, -0.5, 0.5))
                adjustments.append((attr_key, delta, reason))
            else:
                print(f"警告: 不明な属性 '{attr_key}' を無視します")

            if len(adjustments) >= max_suggestions:
                break

        return adjustments
    
    def apply_repulsion(
        self,
        vector: AttributeVector,
        closed_vectors: List[AttributeVector],
        min_squared_distance: float = 0.5,
        repulsion_strength: float = 0.3
    ) -> AttributeVector:
        """
        クローズドノードからの斥力を適用してベクトルを調整
        
        Args:
            vector: 調整対象のベクトル
            closed_vectors: クローズドノードのベクトルリスト
            min_squared_distance: 最小距離の2乗（これより近い場合は斥力を適用）
            repulsion_strength: 斥力の強さ（0.0-1.0）
        
        Returns:
            斥力適用後のベクトル
        """
        if not closed_vectors:
            return vector
        
        adjusted_weights = dict(vector.weights)
        
        for closed_vector in closed_vectors:
            squared_dist = vector.squared_distance(closed_vector)
            
            if squared_dist < min_squared_distance:
                # 距離が近すぎる場合は斥力を適用
                print(f"  斥力適用: 距離の2乗 {squared_dist:.4f} < {min_squared_distance}")
                
                # 差分ベクトルを計算（遠ざける方向）
                all_keys = set(vector.weights.keys()) | set(closed_vector.weights.keys())
                for key in all_keys:
                    current_w = adjusted_weights.get(key, 0.0)
                    closed_w = closed_vector.weights.get(key, 0.0)
                    
                    # 差分の方向に調整（closed_vectorから遠ざける）
                    diff = current_w - closed_w
                    adjustment = repulsion_strength * diff
                    
                    # 新しい重みを計算（0.0～1.0の範囲にクリップ）
                    new_w = np.clip(current_w + adjustment, 0.0, 1.0)
                    adjusted_weights[key] = new_w
        
        return AttributeVector(weights=adjusted_weights)
