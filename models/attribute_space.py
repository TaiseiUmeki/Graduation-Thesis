"""
属性空間定義モジュール
システムで扱う全ての属性と特徴ベクトルの管理
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import json


# # 属性グループの定義
# ATTRIBUTE_GROUPS = {
#     "material": {
#         "stone": "石",
#         "clay_素焼き": "土（素焼き）",
#         "clay_陶器": "土（陶器）",
#         "clay_縄文": "土（縄文土器風）",
#         "metal_iron": "金属（鉄）",
#         "metal_steel": "金属（鋼）",
#         "metal_aluminum": "金属（アルミ）",
#         "metal_copper": "金属（銅）",
#         "metal_brass": "金属（真鍮）",
#         "metal_bronze": "金属（青銅）",
#         "wood_solid": "木（無垢）",
#         "wood_plywood": "木（合板）",
#         "wood_bamboo": "木（竹）",
#         "wood_hinoki": "木（檜）",
#         "wood_oak": "木（オーク）",
#         "wood_walnut": "木（ウォルナット）",
#         "leather_tanned": "革（なめし革）",
#         "leather_suede": "革（スエード）",
#         "fabric_cotton": "布（綿）",
#         "fabric_linen": "布（麻）",
#         "fabric_felt": "布（フェルト）",
#         "fabric_canvas": "布（キャンバス）",
#         "fabric_silk": "布（シルク）",
#         "plastic_abs": "プラスチック（ABS）",
#         "plastic_pp": "プラスチック（PP）",
#         "plastic_pe": "プラスチック（PE）",
#         "plastic_pet": "プラスチック（PET）",
#         "plastic_acrylic": "プラスチック（アクリル）",
#         "plastic_polycarbonate": "プラスチック（ポリカ）",
#         "glass_clear": "ガラス（透明）",
#         "glass_translucent": "ガラス（半透明）",
#         "glass_frosted": "ガラス（フロスト）",
#         "resin_epoxy": "樹脂（エポキシ）",
#         "resin_uv": "樹脂（UVレジン）",
#         "rubber_silicone": "ゴム（シリコン）",
#         "rubber_natural": "ゴム（天然ゴム）",
#         "rubber_foam": "ゴム（発泡ゴム）",
#         "foam_eva": "発泡体（EVA）",
#         "foam_polystyrene": "発泡体（発泡スチロール）",
#         "paper_washi": "紙（和紙）",
#         "paper_cardboard": "紙（厚紙）",
#         "paper_corrugated": "紙（段ボール）",
#         "composite_carbon": "複合材（カーボン）",
#         "composite_frp": "複合材（FRP）",
#     },
    
#     "finish": {
#         "weathered": "風化した",
#         "polished": "磨かれた",
#         "mirror": "鏡面",
#         "hairline": "ヘアライン",
#         "satin": "サテン",
#         "matte": "マット",
#         "matte_finish": "つや消し",
#         "glossy": "艶あり",
#         "hand_formed_traces": "手捻り跡",
#         "fingerprints": "指跡",
#         "chisel_marks": "鑿跡",
#         "carving": "彫り込み",
#         "hammered": "槌目",
#         "cast_surface": "鋳肌",
#         "sandy": "砂目",
#         "rough_cut": "荒削り",
#         "grain_emphasized": "木目強調",
#         "fired": "焼き締め",
#         "painted_solid": "ペイント（単色）",
#         "painted_brushed": "ペイント（刷毛跡あり）",
#         "painted_aged": "ペイント（剥離エイジング）",
#         "dyed": "染色",
#         "smoked": "燻し",
#         "oxidized": "酸化被膜",
#         "rusted": "錆び風",
#         "coating_clear": "コーティング（クリア）",
#         "coating_lacquer": "コーティング（ラッカー）",
#         "coating_oil": "コーティング（オイル）",
#         "coating_wax": "コーティング（ワックス）",
#         "water_repellent": "撥水",
#         "frosted": "フロスト",
#         "pear_skin": "梨地",
#     },
    
#     "shape": {
#         "rotational_symmetry": "回転対称形",
#         "plane_symmetry": "面対称",
#         "point_symmetry": "点対称",
#         "line_symmetry": "線対称",
#         "asymmetric": "非対称",
#         "slender": "細長い",
#         "stout": "ずんぐり",
#         "flat": "扁平",
#         "thin": "薄い",
#         "thick": "厚みがある",
#         "tapered": "テーパー",
#         "stepped": "段付き",
#         "ribbed": "リブ入り",
#         "pointed": "尖っている",
#         "blunt": "鈍頭",
#         "rounded_small": "角の丸み（小R）",
#         "rounded_large": "角の丸み（大R）",
#         "chamfered": "面取り",
#         "sharp_edge": "エッジが立っている",
#         "rounded_corner": "角が落ちている",
#         "sphere": "球",
#         "cube": "立方体",
#         "rectangular": "直方体",
#         "cylinder": "円柱",
#         "cone": "円錐",
#         "pyramid": "角錐",
#         "torus": "トーラス",
#         "capsule": "カプセル",
#         "droplet": "滴形",
#         "leaf": "葉っぱ形",
#         "wavy": "波打ち",
#         "wrinkled": "しわ",
#         "teardrop": "しずく",
#         "streamlined": "流線型",
#     },
    
#     "size": {
#         "extra_small": "極小",
#         "small": "小さい",
#         "medium": "中くらい",
#         "large": "大きい",
#         "extra_large": "特大",
#         "high_aspect_ratio": "細長比が高い",
#         "low_aspect_ratio": "細長比が低い",
#         "lightweight": "軽やか",
#         "solid": "詰まっている",
#         "hollow": "空洞感",
#         "thick_walled": "肉厚",
#     },
    
#     "color": {
#         "cool_blue": "寒色（青）",
#         "cool_cyan": "寒色（青緑）",
#         "cool_indigo": "寒色（藍）",
#         "warm_red": "暖色（赤）",
#         "warm_orange": "暖色（橙）",
#         "warm_yellow": "暖色（黄）",
#         "neutral_gray": "中性色（グレー）",
#         "neutral_beige": "中性色（ベージュ）",
#         "low_saturation": "低彩度",
#         "high_saturation": "高彩度",
#         "monochrome": "モノトーン",
#         "pastel": "パステル",
#         "vivid": "ビビッド",
#         "complementary": "補色配色",
#         "tone_on_tone": "トーン・オン・トーン",
#         "muted": "くすみ色",
#         "japanese_indigo": "和色（藍色）",
#         "japanese_madder": "和色（茜色）",
#         "japanese_matcha": "和色（抹茶色）",
#         "metallic_silver": "金属色（銀）",
#         "metallic_brass": "金属色（真鍮）",
#         "metallic_copper": "金属色（銅）",
#         "translucent": "半透明色",
#         "gradient": "グラデーション",
#     },
    
#     "texture": {
#         "smooth": "さらさら",
#         "moist": "しっとり",
#         "rough": "ざらざら",
#         "slippery": "つるつる",
#         "slimy": "ぬめり",
#         "non_sticky": "ベタつき無し",
#         "elastic": "弾性がある",
#         "soft": "柔らかい",
#         "hard": "硬い",
#         "flexible": "しなる",
#         "cushioned": "クッション性",
#         "grippy": "グリップ感",
#     },
    
#     "structure": {
#         "monolithic": "一体成形",
#         "snap_fit": "はめ合わせ",
#         "crimped": "かしめ",
#         "inserted": "差し込み",
#         "covered": "被せ",
#         "modular": "分割・モジュール化",
#         "stackable": "スタック可能",
#         "foldable": "折りたたみ",
#         "hollow_core": "中空",
#         "honeycomb": "ハニカム",
#         "rib_reinforced": "リブ補強",
#         "with_spacer": "スペーサ",
#         "with_cover": "カバー",
#         "with_casing": "ケーシング",
#     },
    
#     "function": {
#         "rotatable": "回転する",
#         "pushable": "押せる",
#         "pullable": "引ける",
#         "with_handle": "持ち手がある",
#         "hangable": "掛けられる",
#         "clippable": "挟める",
#         "pierceable": "貫ける",
#         "suspendable": "吊れる",
#         "fixable": "固定できる",
#         "screwable": "回して締める",
#         "handle": "取っ手",
#         "knob": "ノブ",
#         "lever": "レバー",
#         "switch_like": "スイッチ風",
#         "button_like": "ボタン風",
#         "hook": "フック",
#         "hinge": "ヒンジ",
#         "slide": "スライド",
#         "lock_mechanism": "ロック機構風",
#         "sound_making": "音が鳴る",
#         "glowing": "光る風",
#         "soft_deformable": "柔らかく変形",
#     },
    
#     "style": {
#         "ancient": "古代風",
#         "jomon": "縄文",
#         "greek": "ギリシャ",
#         "medieval": "中世",
#         "japanese_wabi_sabi": "和風（侘び寂び）",
#         "folk_art": "民芸",
#         "nordic": "北欧",
#         "retro": "レトロ",
#         "showa": "昭和",
#         "industrial": "インダストリアル",
#         "mid_century": "ミッドセンチュリー",
#         "cyber": "サイバー",
#         "futuristic": "近未来",
#         "minimal": "ミニマル",
#         "naive": "素朴",
#         "geometric": "幾何学的",
#         "organic": "有機的",
#         "rugged": "無骨",
#         "elegant": "上品",
#         "cute": "かわいい",
#         "brave": "勇ましい",
#     },
    
#     "visual": {
#         "luxurious": "高級感",
#         "not_cheap": "チープに見せない",
#         "strong_presence": "存在感あり",
#         "modest": "控えめ",
#         "heavy": "重厚",
#         "light": "軽やか",
#         "dynamic": "躍動感",
#         "calm": "落ち着き",
#         "strong_shadow": "影強め",
#         "weak_shadow": "影弱め",
#         "high_contrast": "コントラスト高",
#         "low_contrast": "コントラスト低",
#         "gloss_matte_contrast": "グロス／マットの対比",
#         "aged": "エイジング表現",
#     },
    
#     "pattern": {
#         "grid": "格子",
#         "stripe": "ストライプ",
#         "dot": "ドット",
#         "herringbone": "ヘリンボーン",
#         "leaf": "葉",
#         "wood_grain": "木目",
#         "stone_grain": "石目",
#         "water_flow": "水流",
#         "sand_ripple": "砂紋",
#         "shell": "貝",
#         "ethnic": "民族柄",
#         "japanese_seigaiha": "和柄（青海波）",
#         "japanese_asanoha": "和柄（麻の葉）",
#         "inlay": "象嵌風",
#         "metalwork": "彫金風",
#     }
# }

# 属性グループの定義 (Refined Version)
#ATTRIBUTE_GROUPS = {
    # ---------------------------------------------------------
    # 1. MATERIAL (素材)
    # 物理的な材質。SUN AttributeのMaterialsカテゴリを拡充
    # ---------------------------------------------------------
    # "material": {
    #     # Stone / Earth
    #     "stone_natural": "石（天然石）",
    #     "stone_marble": "石（大理石）",
    #     "stone_concrete": "コンクリート",
    #     "clay_terracotta": "土（素焼き/テラコッタ）",
    #     "clay_ceramic": "土（陶器/セラミック）",
    #     "clay_jomon": "土（縄文土器風）",
        
    #     # Metal
    #     "metal_iron": "金属（鉄/アイアン）",
    #     "metal_steel": "金属（鋼/スチール）",
    #     "metal_aluminum": "金属（アルミ）",
    #     "metal_copper": "金属（銅）",
    #     "metal_brass": "金属（真鍮）",
    #     "metal_bronze": "金属（青銅/ブロンズ）",
        
    #     # Wood
    #     "wood_solid": "木（無垢材）",
    #     "wood_plywood": "木（合板）",
    #     "wood_grain_visible": "木（木目強調）",
    #     "wood_dark_walnut": "木（ウォルナット/暗い）",
    #     "wood_light_oak": "木（オーク/明るい）",
        
    #     # Soft Materials
    #     "leather_aged": "革（なめし革/エイジング）",
    #     "leather_suede": "革（スエード/起毛）",
    #     "fabric_canvas": "布（キャンバス/帆布）",
    #     "fabric_linen": "布（リネン/麻）",
    #     "fabric_tech": "布（テック系/ナイロン）",
        
    #     # Synthetics / Glass
    #     "plastic_glossy": "プラスチック（光沢ABS）",
    #     "plastic_matte": "プラスチック（梨地）",
    #     "plastic_clear": "アクリル/透明樹脂",
    #     "glass_clear": "ガラス（透明）",
    #     "glass_frosted": "ガラス（曇り/フロスト）",
    #     "resin_translucent": "樹脂（半透明）",
    #     "rubber_matte": "ゴム（マット）",
        
    #     # Composites
    #     "composite_carbon_fiber": "カーボンファイバー",
    #     "paper_washi": "和紙",
    #     "paper_cardboard": "クラフト紙/段ボール",
    # },

    # ---------------------------------------------------------
    # 2. SURFACE & FINISH (表面処理)
    # ---------------------------------------------------------
    # "finish": {
    #     # --- A. Reflectivity & Sheen (光沢・反射) ---
    #     "polished_mirror": "鏡面仕上げ（ポリッシュ）",
    #     "satin": "サテン（半光沢）",
    #     "matte_smooth": "マット（平滑）",
    #     "matte_rough": "マット（ザラザラ/梨地）",
    #     "glowing": "発光/自己発光",

    #     # --- B. Manufacturing Texture (加工痕・テクスチャ) ---
    #     "brushed_hairline": "ヘアライン加工",
    #     "hammered": "槌目（ハンマートーン）",
    #     "cast_texture": "鋳肌（キャスト）",
    #     "hand_carved": "手彫り跡/ノミ跡",
    #     "3d_printed_layer": "積層痕（3Dプリント）",

    #     # --- C. Glaze & Vitreous Layers (釉薬・ガラス質・厚み) 【新規追加】 ---
    #     # 陶器や琺瑯、厚みのあるコーティング表現用
    #     "glazed": "施釉（釉薬仕上げ）",
    #     "vitreous": "ガラス質/ビトレアス",  # 硬質で深い透明感
    #     "thick_coating": "厚塗り（ぽってり感）",
    #     "pooling": "液溜まり/釉溜まり",      # 凹部に液体が溜まった表現
    #     "crackle": "貫入（細かいヒビ割れ）", 

    #     # --- D. Paint & Chemical Treatment (塗装・化学処理) ---
    #     "painted_solid": "塗装（単色塗りつぶし）",
    #     "painted_chipped": "塗装剥げ（チッピング）",
    #     "anodized": "アルマイト処理",
    #     "smoked": "燻し加工",

    #     # --- E. Aging & Damage (経年変化・ダメージ) ---
    #     "weathered": "風化/ウェザリング",
    #     "rusted": "錆び（Rust）",
    #     "patina": "緑青/経年変色",
    #     "scratched": "ひっかき傷/スクラッチ",
    # },

    # ---------------------------------------------------------
    # 3. SHAPE & GEOMETRY (形状)
    # プリミティブと変形特徴
    # ---------------------------------------------------------
    # "shape": {
    #     # Primitives
    #     "cube_box": "箱型/立方体",
    #     "cylinder": "円柱",
    #     "sphere_orb": "球体",
    #     "cone": "円錐",
    #     "plate_flat": "板状/フラット",
    #     "organic_blob": "有機的な塊",
        
    #     # Modifiers
    #     "rounded_edges": "角丸（フィレット）",
    #     "chamfered_edges": "面取り（C面）",
    #     "sharp_edges": "ピン角/エッジ重視",
    #     "tapered": "先細り（テーパー）",
    #     "slender": "細長い/スリム",
    #     "chunky": "ずんぐり/塊感",
    #     "hollow": "中空/空洞",
    #     "perforated": "多孔/パンチング",
        
    #     # Complexity
    #     "minimal_simple": "単純形状",
    #     "complex_detailed": "複雑/ディテール過多",
    #     "symmetrical": "対称（シンメトリー）",
    #     "asymmetrical": "非対称（アシンメトリー）",
    #     "streamlined": "流線型",
    # },

    # ---------------------------------------------------------
    # 4. COLOR & TONE (色と調子)
    # 具体的な色相と配色のルール
    # ---------------------------------------------------------
    # "color": {
    #     # Temperature / Tone
    #     "monochrome": "モノトーン（白黒灰）",
    #     "earth_tone": "アースカラー（茶・緑・ベージュ）",
    #     "pastel_tone": "パステルカラー（淡い）",
    #     "vivid_neon": "ビビッド/ネオンカラー",
    #     "muted_desaturated": "低彩度/くすみ色",
    #     "dark_moody": "暗色/重厚",
        
    #     # Specific Schemes
    #     "warm_colors": "暖色系（赤・橙）",
    #     "cool_colors": "寒色系（青・シアン）",
    #     "metallic_silver": "シルバー系",
    #     "metallic_gold": "ゴールド/真鍮系",
    #     "gradient": "グラデーション",
    #     "accent_color": "アクセントカラーあり",
    # },

    # ---------------------------------------------------------
    # 5. FUNCTIONAL PARTS (機能部品・アフォーダンス)
    # 視覚的に機能を示唆するパーツ
    # ---------------------------------------------------------
    # "function_visual": {
    #     "handle_grip": "取っ手/グリップ",
    #     "knob_dial": "つまみ/ダイヤル",
    #     "switch_button": "スイッチ/物理ボタン",
    #     "screen_display": "画面/ディスプレイ",
    #     "vent_slits": "通気口/スリット",
    #     "screws_bolts": "ネジ/ボルト露出",
    #     "modular_joint": "接合部/ジョイント",
    #     "wheels_casters": "車輪/キャスター",
    #     "hinge": "ヒンジ/蝶番",
    # },

    # ---------------------------------------------------------
    # 6. PATTERN & TEXTURE (柄・模様・テクスチャ)
    # ---------------------------------------------------------
    # "pattern": {
    #     # --- A. Geometric & Regular (幾何学・規則的) ---
    #     "stripe_vertical": "ストライプ（縦縞）",
    #     "stripe_horizontal": "ボーダー（横縞）",
    #     "grid_mesh": "グリッド/方眼/メッシュ",
    #     "checkered": "市松模様/チェッカーフラッグ",
    #     "dot_polka": "水玉（ポルカドット）",
    #     "dot_halftone": "ハーフトーン（網点）",
    #     "geometric_hex": "ハニカム（六角形）",
    #     "geometric_triangle": "三角形パターン/ポリゴン",
    #     "herringbone": "ヘリンボーン（杉綾）",
    #     "chevron": "シェブロン（山型）",
    #     "houndstooth": "千鳥格子",
    #     "argyle": "アーガイル（ダイヤ柄）",

        # --- B. Industrial & Functional (工業的・機能的) ---
#         # 表面加工によって生まれる機能的なパターン
#         "knurling_diamond": "ローレット（綾目/ダイヤカット）",
#         "knurling_straight": "ローレット（平目/ストレート）",
#         "carbon_fiber_twill": "カーボン目（綾織）",
#         "carbon_fiber_plain": "カーボン目（平織）",
#         "perforated_hole": "パンチングメタル（丸穴）",
#         "checker_plate": "縞鋼板（チェッカープレート）",
#         "camo_military": "迷彩/カモフラージュ",
#         "camo_digital": "デジタル迷彩",
#         "circuit_board": "回路図パターン/配線",
#         "caution_stripe": "トラ柄（警戒色）",

#         # --- C. Organic & Natural (有機的・自然物) ---
#         "organic_wood_grain": "木目調（プリント/フェイク）",
#         "organic_marble": "マーブル/大理石模様",
#         "organic_flow": "流体/流線模様",
#         "botanical_floral": "花柄/ボタニカル",
#         "animal_leopard": "ヒョウ柄",
#         "animal_zebra": "ゼブラ柄",
#         "terrazzo_speckled": "テラゾー/人造大理石（斑点）",
#         "noise_grain": "ノイズ/砂目",

#         # --- D. Japanese Traditional (和柄) ---
#         # デザインのアクセントとして具体的な名称で定義
#         "wagara_seigaiha": "青海波（せいがいは）",
#         "wagara_asanoha": "麻の葉（あさのは）",
#         "wagara_shippo": "七宝（しっぽう）",
#         "wagara_yagasuri": "矢絣（やがすり）",
#         "wagara_ichimatsu": "市松（和風コンテキスト）",
#         "wagara_karakusa": "唐草（からくさ）",
#         "wagara_kikko": "亀甲（きっこう）",

#         # --- E. Decorative Techniques (装飾技法的な表現) ---
#         "inlay_work": "象嵌（インレイ）風",
#         "damascus_steel": "ダマスカス鋼模様",
#         "mosaic_tile": "モザイクタイル",
#         "paisley": "ペイズリー",
#         "arabesque": "アラベスク/唐草模様",
#         "gradient_fade": "グラデーション/フェード",
#         "typography_logo": "タイポグラフィ/ロゴ配置",
#     }
# }

# 属性グループの定義 (Univocal Geometry Version)
# 感情語・抽象語を排除し、物理的な形状記述語のみに厳選
ATTRIBUTE_GROUPS = {
    # ---------------------------------------------------------
    # 1. FORM FACTOR (基本形態)
    # 物体の最も基礎的な構造分類
    # ---------------------------------------------------------
    "form": {
        # Geometry
        "cubic": "立方体・箱型",
        "cylindrical": "円柱状",
        "spherical": "球状",
        "conical": "円錐状",
        "planar": "板状・平面的",
        
        # Structure
        "hollow": "中空（内部が空洞）",
        "solid": "中実（内部が詰まっている）",
        "mesh": "網状・格子状",
        "frame": "骨組み構造",
        "shell": "殻構造（薄い曲面）",
        
        # Complexity
        "single_mass": "単一塊（継ぎ目なし）",
        "assembly": "複合体（複数の部品の結合）",
    },

    # ---------------------------------------------------------
    # 2. SILHOUETTE & LINE (輪郭と線)
    # 形の印象を決定づける線の性質
    # ---------------------------------------------------------
    "line": {
        # Curvature
        "straight": "直線的",
        "curved": "曲線的",
        "s_curve": "S字カーブ",
        "geometric_curve": "幾何学曲線（正確な円弧など）",
        "organic_curve": "有機的曲線（不規則なうねり）",
        
        # Direction / Flow
        "vertical": "垂直志向",
        "horizontal": "水平志向",
        "radial": "放射状",
        "parallel": "平行的",
        "tapered": "先細り（テーパー）",
        "constricted": "くびれ",
    },

    # ---------------------------------------------------------
    # 3. EDGE & CORNER (エッジと角)
    # 「鋭さ」「柔らかさ」を物理的に定義する
    # ---------------------------------------------------------
    "edge": {
        # Sharpness
        "sharp_angle": "鋭角（ピン角）",
        "right_angle": "直角",
        "obtuse_angle": "鈍角",
        
        # Treatment
        "chamfered": "面取り（平らな削ぎ）",
        "filleted": "角丸（R加工）",
        "rounded_fully": "全体的に丸い",
        "knife_edge": "ナイフエッジ（先端が薄い）",
    },

    # ---------------------------------------------------------
    # 4. SURFACE TOPOLOGY (表面の起伏)
    # 表面がどう変形しているか
    # ---------------------------------------------------------
    "surface": {
        # Protrusion (凸)
        "convex": "凸面（膨らみ）",
        "spiked": "棘状の突起",
        "ribbed": "リブ（畝状の隆起）",
        "embossed": "エンボス（浮き出し）",
        
        # Depression (凹)
        "concave": "凹面（くぼみ）",
        "dimpled": "ディンプル（えくぼ状の穴）",
        "grooved": "溝（スリット）",
        "perforated": "貫通穴",
        
        # Distortion
        "twisted": "ねじれ",
        "bent": "折り曲げ",
        "warped": "歪曲",
        "crumpled": "くしゃくしゃ（不規則な折れ）",
    },

    # ---------------------------------------------------------
    # 5. BALANCE & PROPORTION (均衡と比率)
    # 「安定感」「動き」を数値的に捉えられる概念に分解
    # ---------------------------------------------------------
    "balance": {
        # Symmetry
        "symmetrical": "対称（シンメトリー）",
        "asymmetrical": "非対称（アシンメトリー）",
        
        # Center of Gravity
        "top_heavy": "トップヘビー（上部が大きい）",
        "bottom_heavy": "ボトムヘビー（下部が大きい/安定）",
        
        # Aspect Ratio
        "slender": "細長比が高い（スレンダー）",
        "wide": "幅広",
        "flat": "扁平",
        "thick": "肉厚",
    },

    # ---------------------------------------------------------
    # 6. TEXTURE (触覚的テクスチャ)
    # 視覚的な柄ではなく、触った時の物理形状
    # ---------------------------------------------------------
    "texture": {
        "smooth": "平滑（凹凸なし）",
        "rough": "粗面（不規則なザラつき）",
        "granular": "粒状",
        #"fibrous": "繊維状",
        #"layered": "層状",
        #"cracked": "ひび割れ",
    }, 

    # ---------------------------------------------------------
    # 7. FUNCTIONAL PARTS (機能部品・アフォーダンス)
    # 視覚的に機能を示唆するパーツ
    # ---------------------------------------------------------
    # "function_visual": {
    #     "handle_grip": "取っ手/グリップ",
    #     "knob_dial": "つまみ/ダイヤル",
    #     "switch_button": "スイッチ/物理ボタン",
    #     "screen_display": "画面/ディスプレイ",
    #     "vent_slits": "通気口/スリット",
    #     "screws_bolts": "ネジ/ボルト露出",
    #     "modular_joint": "接合部/ジョイント",
    #     "wheels_casters": "車輪/キャスター",
    #     "hinge": "ヒンジ/蝶番",
    # }
}

@dataclass
class AttributeVector:
    """特徴ベクトルを表すクラス"""
    weights: Dict[str, float]  # 属性名: 重み（0.0～1.0、値が大きいほど強調）
    
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
                vector.weights[attr] = np.clip(weight, 0.0, 1.0)
        return vector
    
    def get_attribute_name(self, attr_key: str) -> Optional[str]:
        """属性キーから日本語名を取得"""
        for group_name, attrs in self.groups.items():
            for key, name in attrs.items():
                if f"{group_name}:{key}" == attr_key:
                    return name
        return None
    
    def has_attribute(self, attr_key: str) -> bool:
        """属性キーが存在するかチェック"""
        return attr_key in self.all_attributes
    
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
