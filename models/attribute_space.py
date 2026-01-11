"""
属性空間定義モジュール
システムで扱う全ての属性と特徴ベクトルの管理
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import json


# 属性グループの定義
ATTRIBUTE_GROUPS = {
    "material": {
        "stone": "石",
        "clay_素焼き": "土（素焼き）",
        "clay_陶器": "土（陶器）",
        "clay_縄文": "土（縄文土器風）",
        "metal_iron": "金属（鉄）",
        "metal_steel": "金属（鋼）",
        "metal_aluminum": "金属（アルミ）",
        "metal_copper": "金属（銅）",
        "metal_brass": "金属（真鍮）",
        "metal_bronze": "金属（青銅）",
        "wood_solid": "木（無垢）",
        "wood_plywood": "木（合板）",
        "wood_bamboo": "木（竹）",
        "wood_hinoki": "木（檜）",
        "wood_oak": "木（オーク）",
        "wood_walnut": "木（ウォルナット）",
        "leather_tanned": "革（なめし革）",
        "leather_suede": "革（スエード）",
        "fabric_cotton": "布（綿）",
        "fabric_linen": "布（麻）",
        "fabric_felt": "布（フェルト）",
        "fabric_canvas": "布（キャンバス）",
        "fabric_silk": "布（シルク）",
        "plastic_abs": "プラスチック（ABS）",
        "plastic_pp": "プラスチック（PP）",
        "plastic_pe": "プラスチック（PE）",
        "plastic_pet": "プラスチック（PET）",
        "plastic_acrylic": "プラスチック（アクリル）",
        "plastic_polycarbonate": "プラスチック（ポリカ）",
        "glass_clear": "ガラス（透明）",
        "glass_translucent": "ガラス（半透明）",
        "glass_frosted": "ガラス（フロスト）",
        "resin_epoxy": "樹脂（エポキシ）",
        "resin_uv": "樹脂（UVレジン）",
        "rubber_silicone": "ゴム（シリコン）",
        "rubber_natural": "ゴム（天然ゴム）",
        "rubber_foam": "ゴム（発泡ゴム）",
        "foam_eva": "発泡体（EVA）",
        "foam_polystyrene": "発泡体（発泡スチロール）",
        "paper_washi": "紙（和紙）",
        "paper_cardboard": "紙（厚紙）",
        "paper_corrugated": "紙（段ボール）",
        "composite_carbon": "複合材（カーボン）",
        "composite_frp": "複合材（FRP）",
    },
    
    "finish": {
        "weathered": "風化した",
        "polished": "磨かれた",
        "mirror": "鏡面",
        "hairline": "ヘアライン",
        "satin": "サテン",
        "matte": "マット",
        "matte_finish": "つや消し",
        "glossy": "艶あり",
        "hand_formed_traces": "手捻り跡",
        "fingerprints": "指跡",
        "chisel_marks": "鑿跡",
        "carving": "彫り込み",
        "hammered": "槌目",
        "cast_surface": "鋳肌",
        "sandy": "砂目",
        "rough_cut": "荒削り",
        "grain_emphasized": "木目強調",
        "fired": "焼き締め",
        "painted_solid": "ペイント（単色）",
        "painted_brushed": "ペイント（刷毛跡あり）",
        "painted_aged": "ペイント（剥離エイジング）",
        "dyed": "染色",
        "smoked": "燻し",
        "oxidized": "酸化被膜",
        "rusted": "錆び風",
        "coating_clear": "コーティング（クリア）",
        "coating_lacquer": "コーティング（ラッカー）",
        "coating_oil": "コーティング（オイル）",
        "coating_wax": "コーティング（ワックス）",
        "water_repellent": "撥水",
        "frosted": "フロスト",
        "pear_skin": "梨地",
    },
    
    "shape": {
        "rotational_symmetry": "回転対称形",
        "plane_symmetry": "面対称",
        "point_symmetry": "点対称",
        "line_symmetry": "線対称",
        "asymmetric": "非対称",
        "slender": "細長い",
        "stout": "ずんぐり",
        "flat": "扁平",
        "thin": "薄い",
        "thick": "厚みがある",
        "tapered": "テーパー",
        "stepped": "段付き",
        "ribbed": "リブ入り",
        "pointed": "尖っている",
        "blunt": "鈍頭",
        "rounded_small": "角の丸み（小R）",
        "rounded_large": "角の丸み（大R）",
        "chamfered": "面取り",
        "sharp_edge": "エッジが立っている",
        "rounded_corner": "角が落ちている",
        "sphere": "球",
        "cube": "立方体",
        "rectangular": "直方体",
        "cylinder": "円柱",
        "cone": "円錐",
        "pyramid": "角錐",
        "torus": "トーラス",
        "capsule": "カプセル",
        "droplet": "滴形",
        "leaf": "葉っぱ形",
        "wavy": "波打ち",
        "wrinkled": "しわ",
        "teardrop": "しずく",
        "streamlined": "流線型",
    },
    
    "size": {
        "extra_small": "極小",
        "small": "小さい",
        "medium": "中くらい",
        "large": "大きい",
        "extra_large": "特大",
        "high_aspect_ratio": "細長比が高い",
        "low_aspect_ratio": "細長比が低い",
        "lightweight": "軽やか",
        "solid": "詰まっている",
        "hollow": "空洞感",
        "thick_walled": "肉厚",
    },
    
    "color": {
        "cool_blue": "寒色（青）",
        "cool_cyan": "寒色（青緑）",
        "cool_indigo": "寒色（藍）",
        "warm_red": "暖色（赤）",
        "warm_orange": "暖色（橙）",
        "warm_yellow": "暖色（黄）",
        "neutral_gray": "中性色（グレー）",
        "neutral_beige": "中性色（ベージュ）",
        "low_saturation": "低彩度",
        "high_saturation": "高彩度",
        "monochrome": "モノトーン",
        "pastel": "パステル",
        "vivid": "ビビッド",
        "complementary": "補色配色",
        "tone_on_tone": "トーン・オン・トーン",
        "muted": "くすみ色",
        "japanese_indigo": "和色（藍色）",
        "japanese_madder": "和色（茜色）",
        "japanese_matcha": "和色（抹茶色）",
        "metallic_silver": "金属色（銀）",
        "metallic_brass": "金属色（真鍮）",
        "metallic_copper": "金属色（銅）",
        "translucent": "半透明色",
        "gradient": "グラデーション",
    },
    
    "texture": {
        "smooth": "さらさら",
        "moist": "しっとり",
        "rough": "ざらざら",
        "slippery": "つるつる",
        "slimy": "ぬめり",
        "non_sticky": "ベタつき無し",
        "elastic": "弾性がある",
        "soft": "柔らかい",
        "hard": "硬い",
        "flexible": "しなる",
        "cushioned": "クッション性",
        "grippy": "グリップ感",
    },
    
    "structure": {
        "monolithic": "一体成形",
        "snap_fit": "はめ合わせ",
        "crimped": "かしめ",
        "inserted": "差し込み",
        "covered": "被せ",
        "modular": "分割・モジュール化",
        "stackable": "スタック可能",
        "foldable": "折りたたみ",
        "hollow_core": "中空",
        "honeycomb": "ハニカム",
        "rib_reinforced": "リブ補強",
        "with_spacer": "スペーサ",
        "with_cover": "カバー",
        "with_casing": "ケーシング",
    },
    
    "function": {
        "rotatable": "回転する",
        "pushable": "押せる",
        "pullable": "引ける",
        "with_handle": "持ち手がある",
        "hangable": "掛けられる",
        "clippable": "挟める",
        "pierceable": "貫ける",
        "suspendable": "吊れる",
        "fixable": "固定できる",
        "screwable": "回して締める",
        "handle": "取っ手",
        "knob": "ノブ",
        "lever": "レバー",
        "switch_like": "スイッチ風",
        "button_like": "ボタン風",
        "hook": "フック",
        "hinge": "ヒンジ",
        "slide": "スライド",
        "lock_mechanism": "ロック機構風",
        "sound_making": "音が鳴る",
        "glowing": "光る風",
        "soft_deformable": "柔らかく変形",
    },
    
    "style": {
        "ancient": "古代風",
        "jomon": "縄文",
        "greek": "ギリシャ",
        "medieval": "中世",
        "japanese_wabi_sabi": "和風（侘び寂び）",
        "folk_art": "民芸",
        "nordic": "北欧",
        "retro": "レトロ",
        "showa": "昭和",
        "industrial": "インダストリアル",
        "mid_century": "ミッドセンチュリー",
        "cyber": "サイバー",
        "futuristic": "近未来",
        "minimal": "ミニマル",
        "naive": "素朴",
        "geometric": "幾何学的",
        "organic": "有機的",
        "rugged": "無骨",
        "elegant": "上品",
        "cute": "かわいい",
        "brave": "勇ましい",
    },
    
    "visual": {
        "luxurious": "高級感",
        "not_cheap": "チープに見せない",
        "strong_presence": "存在感あり",
        "modest": "控えめ",
        "heavy": "重厚",
        "light": "軽やか",
        "dynamic": "躍動感",
        "calm": "落ち着き",
        "strong_shadow": "影強め",
        "weak_shadow": "影弱め",
        "high_contrast": "コントラスト高",
        "low_contrast": "コントラスト低",
        "gloss_matte_contrast": "グロス／マットの対比",
        "aged": "エイジング表現",
    },
    
    "pattern": {
        "grid": "格子",
        "stripe": "ストライプ",
        "dot": "ドット",
        "herringbone": "ヘリンボーン",
        "leaf": "葉",
        "wood_grain": "木目",
        "stone_grain": "石目",
        "water_flow": "水流",
        "sand_ripple": "砂紋",
        "shell": "貝",
        "ethnic": "民族柄",
        "japanese_seigaiha": "和柄（青海波）",
        "japanese_asanoha": "和柄（麻の葉）",
        "inlay": "象嵌風",
        "metalwork": "彫金風",
    }
}


@dataclass
class AttributeVector:
    """特徴ベクトルを表すクラス"""
    weights: Dict[str, float]  # 属性名: 重み（-1.0～1.0、正:強調、負:回避）
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {"weights": self.weights}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AttributeVector':
        """辞書から生成"""
        return cls(weights=data.get("weights", {}))
    
    def to_json(self) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AttributeVector':
        """JSON文字列から生成"""
        return cls.from_dict(json.loads(json_str))
    
    def squared_distance(self, other: 'AttributeVector') -> float:
        """他のベクトルとのユークリッド距離の2乗を計算"""
        all_keys = set(self.weights.keys()) | set(other.weights.keys())
        squared_dist = 0.0
        for key in all_keys:
            w1 = self.weights.get(key, 0.0)
            w2 = other.weights.get(key, 0.0)
            squared_dist += (w1 - w2) ** 2
        return squared_dist


class AttributeSpace:
    """属性空間を管理するクラス"""
    
    def __init__(self):
        self.groups = ATTRIBUTE_GROUPS
        self.all_attributes = self._flatten_attributes()
        self.ndim = len(self.all_attributes)
    
    def _flatten_attributes(self) -> List[str]:
        """全属性をフラットなリストに変換"""
        attributes = []
        for group_name, attrs in self.groups.items():
            for attr_key in attrs.keys():
                attributes.append(f"{group_name}:{attr_key}")
        return attributes
    
    def zero_vector(self) -> AttributeVector:
        """ゼロベクトルを生成"""
        return AttributeVector(weights={attr: 0.0 for attr in self.all_attributes})
    
    def create_vector(self, weights: Dict[str, float]) -> AttributeVector:
        """指定された重みで特徴ベクトルを生成"""
        vector = self.zero_vector()
        for attr, weight in weights.items():
            if attr in self.all_attributes:
                vector.weights[attr] = np.clip(weight, -1.0, 1.0)
        return vector
    
    def get_attribute_name(self, attr_key: str) -> Optional[str]:
        """属性キーから日本語名を取得"""
        for group_name, attrs in self.groups.items():
            for key, name in attrs.items():
                if f"{group_name}:{key}" == attr_key:
                    return name
        return None
    
    def get_group_attributes(self, group_name: str) -> Dict[str, str]:
        """指定グループの属性を取得"""
        return self.groups.get(group_name, {})
    
    def get_top_attributes(self, vector: AttributeVector, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """ベクトルから上位k個の属性を取得（絶対値の大きい順、負値も含む）"""
        sorted_attrs = sorted(
            vector.weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return [(attr, weight) for attr, weight in sorted_attrs[:top_k] if abs(weight) >= threshold]
    
    def vector_to_description(self, vector: AttributeVector, threshold: float = 0.3) -> str:
        """特徴ベクトルを自然言語説明に変換"""
        descriptions = []
        for attr, weight in vector.weights.items():
            if weight >= threshold:
                name = self.get_attribute_name(attr)
                if name:
                    descriptions.append(f"{name}（{weight:.2f}）")
        
        if not descriptions:
            return "特徴なし"
        
        return "、".join(descriptions)
    
    def merge_vectors(self, vector1: AttributeVector, vector2: AttributeVector, 
                     alpha: float = 0.5) -> AttributeVector:
        """2つのベクトルを統合（alpha: vector1の重み）"""
        merged_weights = {}
        for attr in self.all_attributes:
            w1 = vector1.weights.get(attr, 0.0)
            w2 = vector2.weights.get(attr, 0.0)
            merged_weights[attr] = alpha * w1 + (1 - alpha) * w2
        return AttributeVector(weights=merged_weights)


# グローバルインスタンス
ATTR_SPACE = AttributeSpace()
