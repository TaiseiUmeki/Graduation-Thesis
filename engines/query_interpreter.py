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
        previous_interpretations: Optional[List[Interpretation]] = None
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
        system_prompt = self._build_system_prompt()
        
        # ユーザープロンプトの構築
        user_prompt = self._build_user_prompt(
            query,
            num_interpretations,
            previous_interpretations
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
    
    def _build_system_prompt(self) -> str:
        """システムプロンプトを構築"""
        # 属性グループの情報を追加
        attr_groups_info = []
        for group_name, attrs in ATTR_SPACE.groups.items():
            attr_list = list(attrs.values())[:5]  # 各グループから5個だけ例示
            attr_groups_info.append(f"- {group_name}: {', '.join(attr_list)}など")
        
        attr_groups_text = "\n".join(attr_groups_info)
        
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
        previous_interpretations: Optional[List[Interpretation]] = None
    ) -> str:
        """ユーザープロンプトを構築"""
        prompt_parts = [f"ユーザーのクエリ: {query}"]
        
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
      "reasoning": "この解釈に至った理由"
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
                    reasoning=item.get("reasoning", "")
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
