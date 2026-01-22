"""
クエリ解釈エンジン（フェーズB）
ユーザークエリを解釈し、複数の解釈案を生成
"""
from typing import List, Optional
import json

from utils.openai_client import OpenAIClient
from models.session import Interpretation
from models.attribute_space import ATTR_SPACE
from config import Config


class QueryInterpreter:
    """クエリ解釈エンジン"""
    
    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client or OpenAIClient()
    
    def generate_interpretations(
        self,
        query: str,
        num_interpretations: int = None,
        context_image_path: Optional[str] = None,
        previous_interpretations: Optional[List[Interpretation]] = None,
        concept: Optional[str] = None
    ) -> List[Interpretation]:
        """
        クエリから複数の解釈案を生成
        
        Args:
            query: ユーザーのクエリ
            num_interpretations: 生成する解釈案の数
            context_image_path: コンテキスト画像（粘土画像）
            previous_interpretations: 過去の解釈案（詳細化の場合）
        
        Returns:
            解釈案のリスト
        """
        num_interpretations = num_interpretations or Config.NUM_INTERPRETATIONS
        
        # システムプロンプトの構築
        system_prompt = self._build_system_prompt(concept)
        
        # ユーザープロンプトの構築
        user_prompt = self._build_user_prompt(
            query,
            num_interpretations,
            previous_interpretations,
            concept
        )
        
        # 画像がある場合は画像付きで解釈
        if context_image_path:
            response_text = self.client.analyze_image_with_text(
                context_image_path,
                f"{system_prompt}\n\n{user_prompt}"
            )
        else:
            response_text = self.client.chat_completion([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
        
        # レスポンスをパース
        interpretations = self._parse_interpretations(response_text)
        
        return interpretations
    
    def _build_system_prompt(self, concept: Optional[str] = None) -> str:
        """システムプロンプトを構築"""
        # 属性グループの情報を追加
        attr_groups_info = []
        for group_name, attrs in ATTR_SPACE.groups.items():
            attr_list = list(attrs.values())[:5]  # 各グループから5個だけ例示
            attr_groups_info.append(f"- {group_name}: {', '.join(attr_list)}など")
        
        attr_groups_text = "\n".join(attr_groups_info)

        concept_line = "" if not concept else f"対象物: {concept}（ユーザー入力のモチーフ。形状のみを幾何学的に変更）\n"
        
#         return f"""あなたは創作活動を支援するアシスタントです。
# ユーザーは粘土で作った中間生成物の画像を提供し、それをどのように変化させたいかをクエリで伝えます。

# あなたの役割は、ユーザーのクエリを解釈し、複数の異なる解釈案を提示することです。
# 解釈案は具体的で、以下の属性空間を考慮してください：

# {attr_groups_text}

# 各解釈案には以下を含めてください：
# 1. 解釈の内容（具体的にどのような変化を意図しているか）
# 2. その解釈に至った根拠・理由

# 異なる視点や解釈の幅を持たせて、ユーザーが選びやすい選択肢を提供してください。
# """
        return f"""あなたはデザインの専門家であり、創作活動を支援するアシスタントです。
    ユーザーは粘土で作った中間生成物の画像を提供し、それを「形状として」どう変化させたいかをクエリで伝えます。
    {concept_line}

あなたの役割は、ユーザーのクエリを解釈し、形状操作に関する複数の異なる解釈案を提示することです。

【重要：解釈の制約】
1. **形状（Geometry）に限定する**:
   解釈は必ず「形」「輪郭」「ボリューム」「バランス」「エッジ」の変更に留めてください。
2. **材質・表面加工は無視する**:
   「金属にする」「木目をつける」「ピカピカに磨く」「色を変える」といった、材質変更や表面仕上げ（レンダリング的な質感）に関する解釈は**禁止**です。
3. **物理的な操作として捉える**:
   ユーザーの言葉を、粘土に対する物理アクション（削る、盛る、ねじる、引き伸ばす、角を落とす、膨らませる等）として翻訳してください。

以下の属性空間の言葉を用いて、具体的に記述してください：

{attr_groups_text}

各解釈案には以下を含めてください：
1. 解釈の内容（「全体をねじって有機的にする」「エッジを立てて幾何学的にする」など、具体的な形状変化）
2. その解釈に至った根拠・理由（「○○という言葉から、鋭利な形状を連想したため」など）

異なる視点や解釈の幅を持たせて、ユーザーが選びやすい選択肢を提供してください。
"""
    
    def _build_user_prompt(
        self,
        query: str,
        num_interpretations: int,
        previous_interpretations: Optional[List[Interpretation]] = None,
        concept: Optional[str] = None
    ) -> str:
        """ユーザープロンプトを構築"""
        prompt_parts = []
        if concept:
            prompt_parts.append(f"対象物（ユーザー入力）: {concept}")
        prompt_parts.append(f"ユーザーのクエリ: {query}")
        
        if previous_interpretations:
            prompt_parts.append("\n過去の解釈案:")
            for interp in previous_interpretations[-3:]:  # 直近3つ
                prompt_parts.append(f"- {interp.text}")
            prompt_parts.append("\nこれらの解釈案ではユーザーの意図が十分に伝わらなかったようです。")
            prompt_parts.append("より具体的で多様な解釈案を提示してください。")
        
        prompt_parts.append(f"\n{num_interpretations}つの異なる解釈案を以下のJSON形式で提供してください：")
        prompt_parts.append("""
{
  "interpretations": [
    {
      "id": 1,
      "text": "解釈案の内容",
      "reasoning": "この解釈に至った理由",
      "motif": "Tulip" または null （クエリから抽出された固有名詞・モチーフ。英語で。例: Tulip, Cat, Rose。なければnull）
    },
    ...
  ]
}
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_interpretations(self, response_text: str) -> List[Interpretation]:
        """レスポンステキストから解釈案をパース"""
        try:
            # JSONとして解析を試みる
            if "```json" in response_text:
                # マークダウンのコードブロックから抽出
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            data = json.loads(json_text)
            
            interpretations = []
            for item in data.get("interpretations", []):
                interp = Interpretation(
                    id=item["id"],
                    text=item["text"],
                    reasoning=item.get("reasoning", ""),
                    motif=item.get("motif")
                )
                interpretations.append(interp)
            
            return interpretations
        
        except Exception as e:
            print(f"警告: JSON解析に失敗しました: {e}")
            print(f"レスポンス: {response_text}")
            
            # フォールバック: テキストを分割して解釈案を作成
            lines = response_text.split("\n")
            interpretations = []
            current_id = 1
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("{"):
                    interp = Interpretation(
                        id=current_id,
                        text=line,
                        reasoning="自動生成"
                    )
                    interpretations.append(interp)
                    current_id += 1
                    
                    if len(interpretations) >= Config.NUM_INTERPRETATIONS:
                        break
            
            return interpretations
    
    def analyze_concept(
        self,
        image_path: str,
        user_hint: Optional[str] = None
    ) -> dict:
        """
        画像から物体のコンセプトを特定
        
        Args:
            image_path: 分析する画像のパス
            user_hint: ユーザーからのヒント（任意）
        
        Returns:
            {
                "concept_en": "Tank",
                "concept_ja": "戦車",
                "reasoning": "分析の根拠"
            }
        """
        prompt = """この画像に写っている物体が何であるかを特定してください。

以下のJSON形式で回答してください：
{
  "concept_en": "英語の物体名（単数形、画像生成プロンプト用）",
  "concept_ja": "日本語の物体名",
  "reasoning": "この判断に至った根拠"
}

例：
- 戦車のような形をしている → {"concept_en": "Tank", "concept_ja": "戦車", "reasoning": "砲塔と履帯のような構造から"}
- 椅子のような形 → {"concept_en": "Chair", "concept_ja": "椅子", "reasoning": "座面と背もたれが確認できるため"}
"""
        
        if user_hint:
            prompt += f"\n\nユーザーからのヒント: {user_hint}"
        
        response = self.client.analyze_image_with_text(image_path, prompt)
        
        # JSONをパース
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            else:
                json_text = response
            
            result = json.loads(json_text)
            return result
        except Exception as e:
            print(f"警告: JSON解析に失敗しました: {e}")
            print(f"レスポンス: {response}")
            # フォールバック
            return {
                "concept_en": "Object",
                "concept_ja": "物体",
                "reasoning": "自動判定に失敗しました"
            }
    
    def refine_query(
        self,
        original_query: str,
        additional_details: str,
        selected_interpretation: Optional[Interpretation] = None
    ) -> str:
        """
        クエリを詳細化
        
        Args:
            original_query: 元のクエリ
            additional_details: 追加の詳細
            selected_interpretation: 選択された解釈（あれば）
        
        Returns:
            詳細化されたクエリ
        """
        prompt_parts = [f"元のクエリ: {original_query}"]
        prompt_parts.append(f"追加の詳細: {additional_details}")
        
        if selected_interpretation:
            prompt_parts.append(f"近い解釈: {selected_interpretation.text}")
        
        prompt_parts.append("\nこれらを統合して、より明確で詳細なクエリを1文で作成してください。")
        
        prompt = "\n".join(prompt_parts)
        
        refined = self.client.chat_completion([
            {"role": "user", "content": prompt}
        ])
        
        return refined.strip()
