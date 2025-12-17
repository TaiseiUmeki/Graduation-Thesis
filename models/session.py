"""
セッション管理モジュール
ユーザーのセッション状態を管理
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import json
import uuid
from pathlib import Path

from models.attribute_space import AttributeVector
from models.constraints import Constraint


@dataclass
class Interpretation:
    """解釈案を表すクラス"""
    id: int
    text: str  # 解釈の内容
    reasoning: str  # 解釈の根拠
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class GeneratedImage:
    """生成画像を表すクラス"""
    image_path: str
    prompt: str
    vector: Optional[AttributeVector] = None
    constraints: List['Constraint'] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "prompt": self.prompt,
            "vector": self.vector.to_dict() if self.vector else None,
            "constraints": [c.to_dict() for c in self.constraints],
            "timestamp": self.timestamp.isoformat()
        }


class Session:
    """ユーザーセッションを管理するクラス"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # フェーズA: 入力
        self.initial_image_path: Optional[str] = None
        self.initial_query: Optional[str] = None
        
        # フェーズB: クエリ解釈
        self.query_history: List[str] = []  # クエリの履歴
        self.interpretations: List[Interpretation] = []  # 解釈案の履歴
        self.selected_interpretation: Optional[Interpretation] = None
        self.additional_image_path: Optional[str] = None  # 追加の画像（補足用）
        
        # フェーズC: 特徴ベクトル
        self.current_vector: Optional[AttributeVector] = None
        self.recommended_attributes: List[str] = []  # 推薦された属性
        
        # フェーズD: 画像生成と批評
        self.generated_images: List[GeneratedImage] = []
        self.constraints: List['Constraint'] = []  # 制約の履歴
        
        # 現在のフェーズ
        self.current_phase: str = "A"  # A, B, C, D
    
    def add_query(self, query: str):
        """クエリを追加"""
        self.query_history.append(query)
        self.updated_at = datetime.now()
    
    def add_interpretations(self, interpretations: List[Interpretation]):
        """解釈案を追加"""
        self.interpretations.extend(interpretations)
        self.updated_at = datetime.now()
    
    def select_interpretation(self, interpretation_id: int) -> Optional[Interpretation]:
        """解釈案を選択"""
        for interp in self.interpretations:
            if interp.id == interpretation_id:
                self.selected_interpretation = interp
                self.updated_at = datetime.now()
                return interp
        return None
    
    def set_vector(self, vector: AttributeVector):
        """特徴ベクトルを設定"""
        self.current_vector = vector
        self.updated_at = datetime.now()
    
    def add_constraint(self, constraint: 'Constraint'):
        """制約を追加"""
        self.constraints.append(constraint)
        self.updated_at = datetime.now()
    
    def add_generated_image(self, image: GeneratedImage):
        """生成画像を追加"""
        self.generated_images.append(image)
        self.updated_at = datetime.now()
    
    def get_latest_image(self) -> Optional[GeneratedImage]:
        """最新の生成画像を取得"""
        return self.generated_images[-1] if self.generated_images else None
    
    def get_active_constraints(self) -> List['Constraint']:
        """有効な制約を取得"""
        return [c for c in self.constraints if c.is_active]
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "initial_image_path": self.initial_image_path,
            "initial_query": self.initial_query,
            "query_history": self.query_history,
            "interpretations": [i.to_dict() for i in self.interpretations],
            "selected_interpretation": self.selected_interpretation.to_dict() if self.selected_interpretation else None,
            "additional_image_path": self.additional_image_path,
            "current_vector": self.current_vector.to_dict() if self.current_vector else None,
            "recommended_attributes": self.recommended_attributes,
            "generated_images": [img.to_dict() for img in self.generated_images],
            "constraints": [c.to_dict() for c in self.constraints],
            "current_phase": self.current_phase
        }
    
    def save(self, directory: Path):
        """セッションをファイルに保存"""
        filepath = directory / f"{self.session_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'Session':
        """ファイルからセッションを読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session = cls(session_id=data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session.initial_image_path = data.get("initial_image_path")
        session.initial_query = data.get("initial_query")
        session.query_history = data.get("query_history", [])
        session.additional_image_path = data.get("additional_image_path")
        session.recommended_attributes = data.get("recommended_attributes", [])
        session.current_phase = data.get("current_phase", "A")
        
        # 解釈案の復元
        for interp_data in data.get("interpretations", []):
            interp = Interpretation(
                id=interp_data["id"],
                text=interp_data["text"],
                reasoning=interp_data["reasoning"],
                timestamp=datetime.fromisoformat(interp_data["timestamp"])
            )
            session.interpretations.append(interp)
        
        # 選択された解釈の復元
        if data.get("selected_interpretation"):
            interp_data = data["selected_interpretation"]
            session.selected_interpretation = Interpretation(
                id=interp_data["id"],
                text=interp_data["text"],
                reasoning=interp_data["reasoning"],
                timestamp=datetime.fromisoformat(interp_data["timestamp"])
            )
        
        # 特徴ベクトルの復元
        if data.get("current_vector"):
            session.current_vector = AttributeVector.from_dict(data["current_vector"])
        
        # TODO: 生成画像と制約の復元は後で実装
        
        return session
