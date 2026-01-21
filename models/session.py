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
class ExplorationNode:
    """探索木のノードを表すクラス"""
    node_id: int
    parent_id: Optional[int]
    vector: AttributeVector  # Globalベクトル
    partial_vector: Optional[AttributeVector] = None  # 部分編集用ベクトル
    constraints: List['Constraint']
    generated_image_path: Optional[str] = None
    prompt: Optional[str] = None
    mask_image_path: Optional[str] = None  # 部分編集に使用したマスク
    target_part_name: Optional[str] = None  # 編集対象の部位名
    is_closed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "vector": self.vector.to_dict(),
            "partial_vector": self.partial_vector.to_dict() if self.partial_vector else None,
            "constraints": [c.to_dict() for c in self.constraints],
            "generated_image_path": self.generated_image_path,
            "prompt": self.prompt,
            "mask_image_path": self.mask_image_path,
            "target_part_name": self.target_part_name,
            "is_closed": self.is_closed,
            "timestamp": self.timestamp.isoformat(),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExplorationNode':
        return cls(
            node_id=data["node_id"],
            parent_id=data.get("parent_id"),
            vector=AttributeVector.from_dict(data["vector"]),
            partial_vector=AttributeVector.from_dict(data["partial_vector"]) if data.get("partial_vector") else None,
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            generated_image_path=data.get("generated_image_path"),
            prompt=data.get("prompt"),
            mask_image_path=data.get("mask_image_path"),
            target_part_name=data.get("target_part_name"),
            is_closed=data.get("is_closed", False),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            note=data.get("note", ""),
        )


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
        self.concept: Optional[str] = None  # 物体のコンセプト（例: "Tank", "Chair"）
        
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

        # 探索木
        self.exploration_nodes: List[ExplorationNode] = []
        self.current_node_id: Optional[int] = None
        self._next_node_id: int = 0
        
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
        # 最新の解釈案セットで選択できるよう、逆順で探索する
        for interp in reversed(self.interpretations):
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

    def add_root_node(self, vector: AttributeVector, constraints: List['Constraint'], note: str = "", *, partial_vector: Optional[AttributeVector] = None, mask_image_path: Optional[str] = None, target_part_name: Optional[str] = None) -> int:
        """探索木のルートノードを追加"""
        node = ExplorationNode(
            node_id=self._next_node_id,
            parent_id=None,
            vector=self._clone_vector(vector),
            partial_vector=self._clone_vector(partial_vector) if partial_vector else None,
            constraints=self._clone_constraints(constraints),
            mask_image_path=mask_image_path,
            target_part_name=target_part_name,
            note=note,
        )
        self.exploration_nodes.append(node)
        self.current_node_id = node.node_id
        self._next_node_id += 1
        self.updated_at = datetime.now()
        return node.node_id

    def add_child_node(self, vector: AttributeVector, constraints: List['Constraint'], note: str = "", *, partial_vector: Optional[AttributeVector] = None, mask_image_path: Optional[str] = None, target_part_name: Optional[str] = None) -> int:
        """現在のノードの子ノードを追加"""
        node = ExplorationNode(
            node_id=self._next_node_id,
            parent_id=self.current_node_id,
            vector=self._clone_vector(vector),
            partial_vector=self._clone_vector(partial_vector) if partial_vector else None,
            constraints=self._clone_constraints(constraints),
            mask_image_path=mask_image_path,
            target_part_name=target_part_name,
            note=note,
        )
        self.exploration_nodes.append(node)
        self.current_node_id = node.node_id
        self._next_node_id += 1
        self.updated_at = datetime.now()
        return node.node_id

    def update_current_node_image(self, image_path: str, prompt: Optional[str] = None):
        """現在ノードに生成画像パスを保存"""
        node = self._get_current_node()
        if node:
            node.generated_image_path = image_path
            if prompt:
                node.prompt = prompt
            self.updated_at = datetime.now()

    def mark_closed(self, node_id: int):
        """ノードをクローズドとしてマーク"""
        node = self._find_node(node_id)
        if node:
            node.is_closed = True
            self.updated_at = datetime.now()

    def revert_to_node(self, node_id: int):
        """指定ノードのベクトルと制約に戻す"""
        node = self._find_node(node_id)
        if node:
            self.current_node_id = node.node_id
            self.current_vector = self._clone_vector(node.vector)
            self.constraints = self._clone_constraints(node.constraints)
            self.updated_at = datetime.now()
        else:
            raise ValueError(f"ノード {node_id} が見つかりません")

    def list_nodes(self) -> List[Dict]:
        """ノード一覧を返す"""
        result = []
        for n in self.exploration_nodes:
            result.append({
                "node_id": n.node_id,
                "parent_id": n.parent_id,
                "is_closed": n.is_closed,
                "generated_image": n.generated_image_path,
                "timestamp": n.timestamp.isoformat(),
                "note": n.note,
            })
        return result

    def _find_node(self, node_id: int) -> Optional[ExplorationNode]:
        for n in self.exploration_nodes:
            if n.node_id == node_id:
                return n
        return None

    def _get_current_node(self) -> Optional[ExplorationNode]:
        if self.current_node_id is None:
            return None
        return self._find_node(self.current_node_id)
    
    def get_closed_nodes(self) -> List[ExplorationNode]:
        """クローズド（探索終了）ノードのリストを取得"""
        return [n for n in self.exploration_nodes if n.is_closed]

    def _clone_constraints(self, constraints: List['Constraint']) -> List['Constraint']:
        return [Constraint.from_dict(c.to_dict()) for c in constraints]

    def _clone_vector(self, vector: AttributeVector) -> AttributeVector:
        return AttributeVector.from_dict(vector.to_dict())
    
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
            "concept": self.concept,
            "query_history": self.query_history,
            "interpretations": [i.to_dict() for i in self.interpretations],
            "selected_interpretation": self.selected_interpretation.to_dict() if self.selected_interpretation else None,
            "additional_image_path": self.additional_image_path,
            "current_vector": self.current_vector.to_dict() if self.current_vector else None,
            "recommended_attributes": self.recommended_attributes,
            "generated_images": [img.to_dict() for img in self.generated_images],
            "constraints": [c.to_dict() for c in self.constraints],
            "current_phase": self.current_phase,
            "exploration_nodes": [n.to_dict() for n in self.exploration_nodes],
            "current_node_id": self.current_node_id,
            "next_node_id": self._next_node_id,
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
        session.concept = data.get("concept")
        session.initial_image_path = data.get("initial_image_path")
        session.initial_query = data.get("initial_query")
        session.query_history = data.get("query_history", [])
        session.additional_image_path = data.get("additional_image_path")
        session.recommended_attributes = data.get("recommended_attributes", [])
        session.current_phase = data.get("current_phase", "A")
        session._next_node_id = data.get("next_node_id", 0)
        session.current_node_id = data.get("current_node_id")
        
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
        
        # 生成画像の復元
        for img_data in data.get("generated_images", []):
            session.generated_images.append(
                GeneratedImage(
                    image_path=img_data["image_path"],
                    prompt=img_data["prompt"],
                    vector=AttributeVector.from_dict(img_data["vector"]) if img_data.get("vector") else None,
                    constraints=[Constraint.from_dict(c) for c in img_data.get("constraints", [])],
                    timestamp=datetime.fromisoformat(img_data["timestamp"]),
                )
            )

        # 制約の復元
        for constraint_data in data.get("constraints", []):
            session.constraints.append(Constraint.from_dict(constraint_data))

        # 探索ノードの復元
        for node_data in data.get("exploration_nodes", []):
            session.exploration_nodes.append(ExplorationNode.from_dict(node_data))
        
        return session
