"""
画像生成エンジン（フェーズD）
特徴ベクトルから画像を生成し、批評ループを管理
"""
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from utils.openai_client import OpenAIClient
from utils.image_utils import ImageUtils
from models.attribute_space import AttributeVector, ATTR_SPACE
from models.session import GeneratedImage
from models.constraints import Constraint
from config import Config


class ImageGenerator:
    """画像生成エンジン"""

    # ------------------------------------------------------------------
    # Attribute to Adjective Mapping (Phase D - Global / Differential)
    # Thresholds: (Value, Adjective Phrase)
    # Order: Descending (Largest value first)
    # NOTE: Must cover all keys in ATTR_SPACE.all_attributes.
    # ------------------------------------------------------------------
    PROMPT_MAPPING: Dict[str, List[Tuple[float, str]]] = {
        # --- [FORM] Basic geometry ---
        "form:cubic": [
            (0.85, "strongly cubic, box-like silhouette with flat faces"),
            (0.60, "boxy, cubic overall form"),
            (0.30, "slightly box-like proportions"),
            (0.00, "avoid a boxy cube-like silhouette; prefer non-cubic forms"),
        ],
        "form:cylindrical": [
            (0.85, "strongly cylindrical body, clear round cross-section"),
            (0.60, "cylindrical overall shape"),
            (0.30, "somewhat cylindrical profile"),
            (0.00, "avoid a cylinder-like profile; prefer non-cylindrical forms"),
        ],
        "form:spherical": [
            (0.85, "nearly spherical mass, ball-like volume"),
            (0.60, "rounded spherical overall form"),
            (0.30, "somewhat rounded, sphere-influenced volume"),
            (0.00, "avoid a ball-like spherical mass; prefer non-spherical forms"),
        ],
        "form:conical": [
            (0.85, "distinct conical form, clear taper to a point"),
            (0.60, "conical silhouette with a noticeable taper"),
            (0.30, "slightly conical, subtle tapering"),
            (0.00, "avoid a cone-like taper; prefer non-conical forms"),
        ],
        "form:planar": [
            (0.85, "strongly planar, plate-like, broad flat surfaces"),
            (0.60, "flat, planar form with wide faces"),
            (0.30, "somewhat planar, slightly flattened profile"),
            (0.00, "avoid an overly flat planar plate-like form; prefer volumetric mass"),
        ],
        "form:hollow": [
            (0.85, "clearly hollow interior, pronounced cavity or void"),
            (0.60, "hollow structure with visible internal space"),
            (0.30, "somewhat hollow, hint of internal void"),
            (0.00, "solid filled volume; avoid hollow cavities and voids"),
        ],
        "form:solid": [
            (0.85, "solid monolithic volume, fully filled mass"),
            (0.60, "solid, dense filled form"),
            (0.30, "somewhat solid, reduced hollowness"),
            (0.00, "avoid a fully solid block; allow hollowness and internal voids"),
        ],
        "form:mesh": [
            (0.85, "intricate lattice mesh structure, open grid-like body"),
            (0.60, "mesh-like structure with grid openings"),
            (0.30, "light perforations or sparse mesh quality"),
            (0.00, "solid surface with no mesh; avoid grid openings and lattice"),
        ],
        "form:frame": [
            (0.85, "exposed skeletal frame, strut-based structure"),
            (0.60, "frame-like construction with visible supports"),
            (0.30, "partly frame-based, some exposed supports"),
            (0.00, "avoid exposed framing; prefer continuous surfaces"),
        ],
        "form:shell": [
            (0.85, "thin shell-like curved surface, lightweight skin"),
            (0.60, "shell structure with thin curved surfaces"),
            (0.30, "somewhat shell-like, partial thin surface skin"),
            (0.00, "avoid thin shell skin; prefer thicker volumetric structure"),
        ],
        "form:single_mass": [
            (0.85, "single uninterrupted mass, no seams or separations"),
            (0.60, "mostly single-piece, continuous mass"),
            (0.30, "somewhat continuous mass with minimal segmentation"),
            (0.00, "assembled from multiple parts; avoid a single uninterrupted mass"),
        ],
        "form:assembly": [
            (0.85, "clearly assembled multi-part structure, distinct components"),
            (0.60, "composed of multiple parts, modular assembly feel"),
            (0.30, "slight multi-part impression"),
            (0.00, "single continuous mass; avoid visible component assembly"),
        ],

        # --- [LINE] Silhouette & flow ---
        "line:straight": [
            (0.85, "dominantly straight lines, rigid linear silhouette"),
            (0.60, "straight-edged linework and linear contours"),
            (0.30, "some straight segments in the outline"),
            (0.00, "avoid straight rigid lines; prefer curved flowing contours"),
        ],
        "line:curved": [
            (0.85, "strongly curved outline, smooth flowing curvature"),
            (0.60, "curved contours and rounded line flow"),
            (0.30, "slightly curved, gentle arcs"),
            (0.00, "avoid excessive curvature; prefer straighter, more linear contours"),
        ],
        "line:s_curve": [
            (0.85, "pronounced S-curve flow, dynamic serpentine silhouette"),
            (0.60, "S-shaped curves and sinuous lines"),
            (0.30, "subtle S-curve accents"),
            (0.00, "avoid S-shaped serpentine flow; prefer simpler single-direction curves"),
        ],
        "line:geometric_curve": [
            (0.85, "precise geometric curves (perfect arcs), engineered curvature"),
            (0.60, "geometric arc-like curves, controlled curvature"),
            (0.30, "slightly geometric, orderly curvature"),
            (0.00, "avoid perfect geometric arcs; prefer irregular organic curvature"),
        ],
        "line:organic_curve": [
            (0.85, "highly organic irregular curves, natural flowing linework"),
            (0.60, "organic curves, irregular flowing contours"),
            (0.30, "slightly organic line quality"),
            (0.00, "geometric and regular linework; avoid organic irregular curves"),
        ],
        "line:vertical": [
            (0.85, "strong vertical emphasis, upright proportions"),
            (0.60, "vertical orientation and upward flow"),
            (0.30, "slight vertical tendency"),
            (0.00, "avoid vertical emphasis; prefer horizontal or radial orientation"),
        ],
        "line:horizontal": [
            (0.85, "strong horizontal emphasis, wide sideways flow"),
            (0.60, "horizontal orientation and lateral spread"),
            (0.30, "slight horizontal tendency"),
            (0.00, "avoid horizontal emphasis; prefer vertical or radial orientation"),
        ],
        "line:radial": [
            (0.85, "strong radial organization, spokes radiating from a center"),
            (0.60, "radial layout with elements radiating outward"),
            (0.30, "slight radial accents"),
            (0.00, "avoid radial organization; prefer parallel or directional flow"),
        ],
        "line:parallel": [
            (0.85, "strong parallel alignment, repeated parallel lines"),
            (0.60, "parallel line rhythm and aligned elements"),
            (0.30, "some parallel alignment"),
            (0.00, "avoid parallel repetition; prefer varied non-parallel directions"),
        ],
        "line:tapered": [
            (0.85, "strong tapering, clearly narrowing toward an end"),
            (0.60, "noticeable taper, narrowing profile"),
            (0.30, "slight tapering"),
            (0.00, "avoid tapering; keep thickness more uniform"),
        ],
        "line:constricted": [
            (0.85, "strong constriction, pronounced waist and pinch"),
            (0.60, "constricted midsection, clear necking-in"),
            (0.30, "slight constriction"),
            (0.00, "avoid a pinched waist; keep cross-section more uniform"),
        ],

        # --- [EDGE] Edge & corner treatment ---
        "edge:sharp_angle": [
            (0.85, "extremely sharp angles, razor-like crisp corners"),
            (0.60, "sharp angular corners, crisp edges"),
            (0.30, "slightly angular corners"),
            (0.00, "rounded edges, soft corners, filleted transitions"),
        ],
        "edge:right_angle": [
            (0.85, "dominant right angles (90°), orthogonal geometry"),
            (0.60, "mostly right-angled corners, orthogonal structure"),
            (0.30, "some right-angled features"),
            (0.00, "avoid strict right angles; prefer oblique angles or rounded corners"),
        ],
        "edge:obtuse_angle": [
            (0.85, "broad obtuse angles, open blunt corners"),
            (0.60, "mostly obtuse-angled corners"),
            (0.30, "slightly obtuse corners"),
            (0.00, "avoid blunt obtuse corners; prefer sharper or more defined angles"),
        ],
        "edge:chamfered": [
            (0.85, "strong chamfers, flat beveled edge cuts"),
            (0.60, "chamfered edges, noticeable bevels"),
            (0.30, "slight chamfering on edges"),
            (0.00, "no chamfers; keep edges either sharp or smoothly filleted"),
        ],
        "edge:filleted": [
            (0.85, "large fillets, generous radius transitions on corners"),
            (0.60, "filleted corners with smooth radius"),
            (0.30, "slightly rounded fillets"),
            (0.00, "crisp corners with minimal rounding; avoid filleted transitions"),
        ],
        "edge:rounded_fully": [
            (0.85, "overall fully rounded, soft blob-like corners everywhere"),
            (0.60, "mostly rounded with soft overall edge treatment"),
            (0.30, "slightly rounded overall"),
            (0.00, "avoid fully rounded blob-like edges; prefer defined corners or bevels"),
        ],
        "edge:knife_edge": [
            (0.85, "knife-edge thin tips, very thin sharp terminating edges"),
            (0.60, "thin knife-like edges and tips"),
            (0.30, "some thin-edged details"),
            (0.00, "avoid thin knife edges; use thicker blunt terminations"),
        ],

        # --- [SURFACE] Surface topology ---
        "surface:convex": [
            (0.85, "strong convex bulges, pronounced swelling surfaces"),
            (0.60, "convex surface bulging outward"),
            (0.30, "slight convex swelling"),
            (0.00, "avoid bulging convexity; keep surfaces flatter or slightly concave"),
        ],
        "surface:spiked": [
            (0.85, "many sharp spikes and thorn-like protrusions"),
            (0.60, "spiky protrusions, pointed surface details"),
            (0.30, "a few small spikes"),
            (0.00, "smooth safe surface; avoid spikes and thorn-like protrusions"),
        ],
        "surface:ribbed": [
            (0.85, "strong ribbing, deep repeated ridges"),
            (0.60, "ribbed surface with noticeable ridges"),
            (0.30, "light ribbing, subtle ridges"),
            (0.00, "no ribbing; avoid repeated ridge patterns"),
        ],
        "surface:embossed": [
            (0.85, "strong embossing, bold raised relief patterns"),
            (0.60, "embossed raised details on the surface"),
            (0.30, "subtle embossed relief"),
            (0.00, "flat surface without embossed relief; avoid raised patterns"),
        ],
        "surface:concave": [
            (0.85, "strong concave hollows, deep inward curving surfaces"),
            (0.60, "concave surface indentations"),
            (0.30, "slight concave dimpling"),
            (0.00, "avoid deep concavities; prefer outward convex or flat surfaces"),
        ],
        "surface:dimpled": [
            (0.85, "dense dimples, many small pit-like dents"),
            (0.60, "dimpled surface with multiple small dents"),
            (0.30, "a few dimples"),
            (0.00, "smooth surface; avoid pitted dimple texture"),
        ],
        "surface:grooved": [
            (0.85, "deep grooves and slits, strong carved channels"),
            (0.60, "grooved surface with noticeable channels"),
            (0.30, "light grooves, subtle channels"),
            (0.00, "no grooves; avoid slit-like channels and carved lines"),
        ],
        "surface:perforated": [
            (0.85, "many through-holes, heavily perforated body"),
            (0.60, "perforated surface with multiple holes"),
            (0.30, "a few perforations"),
            (0.00, "solid surface; no holes or perforations"),
        ],
        "surface:twisted": [
            (0.85, "strong twisting deformation, torsion along the body"),
            (0.60, "twisted form with visible torsion"),
            (0.30, "slight twist"),
            (0.00, "avoid twisting; keep geometry untwisted and aligned"),
        ],
        "surface:bent": [
            (0.85, "strong bending, clearly curved by bending deformation"),
            (0.60, "bent form with a noticeable bend"),
            (0.30, "slight bend"),
            (0.00, "avoid bending; keep form straight and unbent"),
        ],
        "surface:warped": [
            (0.85, "strong warping, distorted and uneven surface flow"),
            (0.60, "warped surface, visibly distorted"),
            (0.30, "slight warping"),
            (0.00, "avoid warping; keep surfaces even and undistorted"),
        ],
        "surface:crumpled": [
            (0.85, "strongly crumpled, irregular folded dents and creases"),
            (0.60, "crumpled surface with irregular folds"),
            (0.30, "slightly crumpled, a few creases"),
            (0.00, "smooth continuous surface; avoid crumples and irregular folds"),
        ],

        # --- [BALANCE] Balance & proportion ---
        "balance:symmetrical": [
            (0.85, "highly symmetrical, mirrored left-right balance"),
            (0.60, "mostly symmetrical overall balance"),
            (0.30, "slightly symmetrical structure"),
            (0.00, "asymmetrical, irregular balance; avoid strict symmetry"),
        ],
        "balance:asymmetrical": [
            (0.85, "strongly asymmetrical, intentionally unbalanced arrangement"),
            (0.60, "asymmetrical overall form"),
            (0.30, "slightly asymmetrical details"),
            (0.00, "symmetrical and evenly balanced; avoid asymmetry"),
        ],
        "balance:top_heavy": [
            (0.85, "prominent oversized upper mass, strongly top-heavy"),
            (0.60, "top-heavy balance with larger upper portion"),
            (0.30, "slightly top-heavy"),
            (0.00, "visually balanced vertical weight distribution, stable and neutral"),
        ],
        "balance:bottom_heavy": [
            (0.85, "wide stable base, massive bottom, strongly bottom-heavy"),
            (0.60, "bottom-heavy balance with weighted base"),
            (0.30, "slightly bottom-weighted"),
            (0.00, "visually balanced vertical weight distribution, straight stable profile"),
        ],
        "balance:slender": [
            (0.85, "very slender, high aspect ratio, tall and thin"),
            (0.60, "slender proportions, elongated"),
            (0.30, "slightly slender"),
            (0.00, "avoid slenderness; prefer broader, lower aspect ratio proportions"),
        ],
        "balance:wide": [
            (0.85, "very wide proportions, low aspect ratio, broad footprint"),
            (0.60, "wide and broad proportions"),
            (0.30, "slightly wide"),
            (0.00, "avoid wide squat proportions; prefer slimmer or taller proportions"),
        ],
        "balance:flat": [
            (0.85, "strongly flattened, thin vertical thickness, low profile"),
            (0.60, "flat, low-profile form"),
            (0.30, "slightly flattened"),
            (0.00, "avoid flatness; prefer thicker, more volumetric height"),
        ],
        "balance:thick": [
            (0.85, "very thick and chunky, substantial thickness"),
            (0.60, "thick, bulky proportions"),
            (0.30, "slightly thick"),
            (0.00, "avoid bulkiness; prefer thinner or lighter proportions"),
        ],

        # --- [TEXTURE] Tactile surface texture ---
        "texture:smooth": [
            (0.85, "very smooth surface, minimal micro-relief"),
            (0.60, "smooth surface with little roughness"),
            (0.30, "slightly smooth surface"),
            (0.00, "roughened surface with visible irregularities; avoid smooth finish"),
        ],
        "texture:rough": [
            (0.85, "very rough surface, strong irregular micro-relief"),
            (0.60, "rough surface with noticeable texture"),
            (0.30, "slightly rough texture"),
            (0.00, "smooth surface; avoid rough irregular micro-relief"),
        ],
        "texture:granular": [
            (0.85, "highly granular surface, many small grains and bumps"),
            (0.60, "granular surface with fine grains"),
            (0.30, "slight graininess"),
            (0.00, "smooth non-granular surface; avoid grainy bumps"),
        ],
    }
    
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
        motif: Optional[str] = None,
        base_interpretation: Optional[str] = None,
        initial_vector: Optional[AttributeVector] = None
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
        prompt = self._build_image_prompt(
            vector,
            constraints,
            concept,
            motif,
            base_interpretation=base_interpretation,
            initial_vector=initial_vector,
            current_vector=vector,
            active_constraints=constraints,
        )
        
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

    def generate_from_prompt(
        self,
        base_image_path: str,
        prompt: str,
        output_dir: Optional[Path] = None,
        *,
        vector: Optional[AttributeVector] = None,
        constraints: Optional[List[Constraint]] = None
    ) -> GeneratedImage:
        """
        参照画像 + 任意のプロンプトで画像を生成（Phase D - Global用）
        """
        output_dir = output_dir or Config.IMAGES_DIR

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

        return GeneratedImage(
            image_path=image_path,
            prompt=prompt,
            vector=vector or AttributeVector(weights={}),
            constraints=constraints or []
        )
    
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
        vector: Optional[AttributeVector] = None,
        constraints: Optional[List[Constraint]] = None,
        concept: Optional[str] = None,
        motif: Optional[str] = None,
        base_interpretation: Optional[str] = None,
        initial_vector: Optional[AttributeVector] = None,
        current_vector: Optional[AttributeVector] = None,
        active_constraints: Optional[List[Constraint]] = None
    ) -> str:
        """
        画像生成用のプロンプトを構築

        - base_interpretation + initial_vector + current_vector が与えられた場合:
          差分駆動（Differential Prompting）で形容詞注入を行う（Phase D - Global）。
        - それ以外:
          既存の属性列挙スタイル（互換）。
        """
        if base_interpretation is not None and initial_vector is not None and current_vector is not None:
            return self._build_image_prompt_differential(
                base_interpretation=base_interpretation,
                initial_vector=initial_vector,
                current_vector=current_vector,
                active_constraints=active_constraints or constraints or [],
                concept=concept,
                motif=motif,
            )

        if vector is None:
            vector = AttributeVector(weights={})

        # ----- legacy prompt -----
        prompt_parts = []

        if concept:
            prompt_parts.append(f"A 3D render of a {concept} made of CLAY.")
        else:
            prompt_parts.append("A 3D render of an object made of CLAY.")

        # if motif:
        #     prompt_parts.append(f"It is designed with the distinct motif of a {motif}.")
        #     prompt_parts.append(f"Incorporate the characteristic shape and silhouette of a {motif} into the design.")

        prompt_parts.append("Keep the entire clay object fully inside the frame with no cropping or cut-off edges; center it with a small margin around the subject.")
        prompt_parts.append("【最重要】Keep the material as raw clay. Only modify the GEOMETRY (shape, edges, contours) based on the following attributes.")
        prompt_parts.append("（注：各属性の値は0.0～1.0の範囲です。値が大きいほどその特徴を強調します）")

        top_attributes = self.attr_space.get_top_attributes(vector, top_k=20, threshold=0.0)

        if top_attributes:
            grouped_attrs = {}
            for attr_key, weight in top_attributes:
                group_name = attr_key.split(':')[0]
                attr_name = self.attr_space.get_attribute_name(attr_key)

                if attr_name:
                    if group_name not in grouped_attrs:
                        grouped_attrs[group_name] = []
                    grouped_attrs[group_name].append((attr_name, weight))

            attr_descriptions = []
            for group_name in self.attr_space.groups.keys():
                if group_name not in grouped_attrs:
                    continue

                attrs_formatted = []
                for name, weight in grouped_attrs[group_name][:Config.MAX_ATTRS_PER_GROUP]:
                    attrs_formatted.append(f"{name}({weight:.2f})")

                if attrs_formatted:
                    attr_descriptions.append(f"{group_name}: {', '.join(attrs_formatted)}")

            if attr_descriptions:
                prompt_parts.append("、".join(attr_descriptions))

        if constraints:
            active_cs = [c for c in constraints if c.is_active]
            if active_cs:
                constraint_texts = [c.description for c in active_cs if c.description]
                if constraint_texts:
                    prompt_parts.append(f"制約: {', '.join(constraint_texts)}")

        return "。".join(prompt_parts) + "。粘土のテクスチャを保持した、高品質で詳細な3Dレンダリング。"

    def _build_image_prompt_differential(
        self,
        base_interpretation: str,
        initial_vector: AttributeVector,
        current_vector: AttributeVector,
        active_constraints: List[Constraint],
        concept: Optional[str] = None,
        motif: Optional[str] = None
    ) -> str:
        """
        差分駆動（Differential Prompting）でプロンプトを構築
        - ユーザーが操作した属性（制約）または値が大きく変化した属性のみを注入
        - 値が低い（0.0付近）場合は、対義語 or ニュートラル表現で上書きを狙う
        """
        prompt_parts: List[str] = []

        obj_name = concept if concept else "object"
        prompt_parts.append(f"A 3D render of a {obj_name} made of CLAY.")

        # if motif:
        #     prompt_parts.append(f"It is designed with the distinct motif of a {motif}.")
        #     prompt_parts.append(f"Incorporate the characteristic shape and silhouette of a {motif} into the design.")

        prompt_parts.append(base_interpretation)

        prompt_parts.append("Keep the entire clay object fully inside the frame with no cropping or cut-off edges; center it with a small margin around the subject.")
        prompt_parts.append("【最重要】Keep the material as raw clay. Only modify the GEOMETRY (shape, edges, contours).")

        constrained_keys = {c.attribute for c in active_constraints if getattr(c, "is_active", True)}
        added_adjectives: List[str] = []
        delta_threshold = 0.2

        for attr_key, mappings in self.PROMPT_MAPPING.items():
            current_val = float(current_vector.weights.get(attr_key, 0.0))
            initial_val = float(initial_vector.weights.get(attr_key, 0.0))

            is_constrained = attr_key in constrained_keys
            is_changed = abs(current_val - initial_val) >= delta_threshold

            if not (is_constrained or is_changed):
                continue

            prompt_text = ""
            for threshold, text in mappings:
                if current_val >= threshold:
                    prompt_text = text
                    break

            if prompt_text:
                added_adjectives.append(prompt_text)

        if added_adjectives:
            distinct_adjectives = list(dict.fromkeys(added_adjectives))
            prompt_parts.append(f"Modified features: {', '.join(distinct_adjectives)}.")

        prompt_parts.append("High quality, detailed clay texture, neutral studio lighting.")

        return " ".join(prompt_parts)
    
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
