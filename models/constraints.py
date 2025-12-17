"""
制約管理モジュール
属性値に対する制約を管理
"""
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum


class ConstraintType(Enum):
    """制約のタイプ"""
    LESS_THAN = "less_than"  # <=
    GREATER_THAN = "greater_than"  # >=
    EQUAL = "equal"  # =
    RANGE = "range"  # min <= x <= max


@dataclass
class Constraint:
    """属性に対する制約を表すクラス"""
    attribute: str  # 属性名（例: "material:wood_oak"）
    constraint_type: ConstraintType
    value: Optional[float] = None  # 単一値の場合
    min_value: Optional[float] = None  # 範囲制約の最小値
    max_value: Optional[float] = None  # 範囲制約の最大値
    description: str = ""  # 制約の説明
    is_active: bool = True  # 制約が有効かどうか
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def validate(self, attr_value: float) -> bool:
        """値が制約を満たすか検証"""
        if not self.is_active:
            return True
        
        if self.constraint_type == ConstraintType.LESS_THAN:
            return attr_value <= self.value
        elif self.constraint_type == ConstraintType.GREATER_THAN:
            return attr_value >= self.value
        elif self.constraint_type == ConstraintType.EQUAL:
            return abs(attr_value - self.value) < 1e-6
        elif self.constraint_type == ConstraintType.RANGE:
            return self.min_value <= attr_value <= self.max_value
        return True
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "attribute": self.attribute,
            "constraint_type": self.constraint_type.value,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
            "is_active": self.is_active,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Constraint':
        """辞書から生成"""
        return cls(
            attribute=data["attribute"],
            constraint_type=ConstraintType(data["constraint_type"]),
            value=data.get("value"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def to_natural_language(self) -> str:
        """制約を自然言語で表現"""
        if self.constraint_type == ConstraintType.LESS_THAN:
            return f"{self.attribute} ≤ {self.value:.2f}"
        elif self.constraint_type == ConstraintType.GREATER_THAN:
            return f"{self.attribute} ≥ {self.value:.2f}"
        elif self.constraint_type == ConstraintType.EQUAL:
            return f"{self.attribute} = {self.value:.2f}"
        elif self.constraint_type == ConstraintType.RANGE:
            return f"{self.min_value:.2f} ≤ {self.attribute} ≤ {self.max_value:.2f}"
        return str(self)


class ConstraintManager:
    """制約を管理するクラス"""
    
    def __init__(self):
        self.constraints: List[Constraint] = []
    
    def add_constraint(self, constraint: Constraint):
        """制約を追加"""
        self.constraints.append(constraint)
    
    def remove_constraint(self, index: int):
        """制約を削除"""
        if 0 <= index < len(self.constraints):
            self.constraints.pop(index)
    
    def deactivate_constraint(self, index: int):
        """制約を無効化"""
        if 0 <= index < len(self.constraints):
            self.constraints[index].is_active = False
    
    def activate_constraint(self, index: int):
        """制約を有効化"""
        if 0 <= index < len(self.constraints):
            self.constraints[index].is_active = True
    
    def get_active_constraints(self) -> List[Constraint]:
        """有効な制約を取得"""
        return [c for c in self.constraints if c.is_active]
    
    def validate_vector(self, vector_weights: Dict[str, float]) -> tuple[bool, List[str]]:
        """
        ベクトルが全制約を満たすか検証
        Returns:
            (満たすかどうか, 違反した制約のリスト)
        """
        violations = []
        for constraint in self.get_active_constraints():
            attr_value = vector_weights.get(constraint.attribute, 0.0)
            if not constraint.validate(attr_value):
                violations.append(constraint.to_natural_language())
        
        return len(violations) == 0, violations
    
    def get_constraints_for_attribute(self, attribute: str) -> List[Constraint]:
        """特定の属性に関する制約を取得"""
        return [c for c in self.constraints if c.attribute == attribute and c.is_active]
    
    def to_prompt_text(self) -> str:
        """制約をプロンプト用のテキストに変換"""
        active = self.get_active_constraints()
        if not active:
            return "制約なし"
        
        lines = ["以下の制約を満たす必要があります："]
        for i, c in enumerate(active, 1):
            lines.append(f"{i}. {c.description or c.to_natural_language()}")
        
        return "\n".join(lines)
