import os
import uuid
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

# ページ設定は最初のStreamlitコマンドとして実行
st.set_page_config(page_title="梅木卒論 GUI", layout="wide")

# APIキー設定（最初にチェック）
if not os.getenv("OPENAI_API_KEY"):
    st.sidebar.warning("⚠️ OpenAI APIキーが設定されていません")
    api_key = st.sidebar.text_input("OpenAI APIキーを入力", type="password", key="api_key_input")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.sidebar.success("APIキーを設定しました")
    else:
        st.error("APIキーを入力してください（左のサイドバーから）")
        st.stop()

# 既存システムの読み込み
from main import TrueCodingSystem
from config import Config
from models.attribute_space import ATTR_SPACE


# ヘルパ: 一時ファイル保存（画像）
def _save_uploaded_image(uploaded_file, prefix: str = "uploaded") -> str:
    if uploaded_file is None:
        return ""
    images_dir = Config.INPUT_IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix.lower() or ".jpg"
    fname = f"{prefix}_{uuid.uuid4().hex}{ext}"
    fpath = images_dir / fname
    with open(fpath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(fpath)

# ヘルパ: 画像を開く（キャッシュ付き）
@st.cache_resource
def _load_image(image_path: str):
    """画像をキャッシュして読み込む（PIL Imageオブジェクトとして）"""
    return Image.open(image_path)

# ヘルパ: 現在の特徴ベクトルの上位をDataFrame化
def _vector_top_df(weights: dict, top_k: int = 10) -> pd.DataFrame:
    items = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    rows = []
    for key, val in items:
        name = ATTR_SPACE.get_attribute_name(key) or key
        rows.append({"属性": name, "キー": key, "値": float(val)})
    df = pd.DataFrame(rows)
    return df

# ヘルパ: 探索木をGraphviz DOTに変換
try:
    import graphviz as gv
except Exception:
    gv = None

def _build_tree_dot(session) -> str:
    nodes = session.exploration_nodes
    lines = ["digraph G {", "rankdir=LR;", "node [shape=box, style=filled, fillcolor=white, fontname=\"Noto Sans CJK JP\"]; "]
    for n in nodes:
        label_parts = [f"#{n.node_id}"]
        if n.note:
            label_parts.append(n.note)
        if n.is_closed:
            label_parts.append("[CLOSED]")
        label = " ".join(label_parts)
        fill = "#ffeeee" if n.is_closed else "#ffffff"
        lines.append(f"n{n.node_id} [label=\"{label}\", fillcolor=\"{fill}\"]; ")
    for n in nodes:
        if n.parent_id is not None:
            lines.append(f"n{n.parent_id} -> n{n.node_id};")
    lines.append("}")
    return "\n".join(lines)

# ヘルパ: 属性空間リファレンス表示（3カラム）
def _display_attribute_reference():
    """属性空間をグループごとに3カラムで表示"""
    for gname, items in ATTR_SPACE.groups.items():
        st.subheader(f"🏷️ {gname}")
        # グループの属性を3カラムに分割
        cols = st.columns(3)
        attr_list = list(items.items())
        for idx, (k, v) in enumerate(attr_list):
            col_idx = idx % 3
            with cols[col_idx]:
                st.caption(f"**{v}**")
                st.code(f"{gname}:{k}", language=None)

# セッション状態初期化
if "system" not in st.session_state:
    st.session_state.system = TrueCodingSystem()
    st.session_state.phase = "A"
    st.session_state.interpretations = []  # (id, text, reasoning)
    st.session_state.selected_interpretation_id = None
    st.session_state.analysis_text = ""
    st.session_state.search_results = []
    # Phase A用の追加状態
    st.session_state.uploaded_image_path = None
    st.session_state.synthesis_result_path = None
    # 部分編集用の状態
    st.session_state.partial_interpretations = []
    st.session_state.selected_partial_interpretation_id = None
    st.session_state.partial_edit_mask_path = None
    st.session_state.partial_edit_target_name = None
    st.session_state.partial_edit_base_path = None
    st.session_state.partial_edit_query = None
    # 合成モード用の状態
    st.session_state.synthesis_base_path = None
    st.session_state.synthesis_mask_path = None
    st.session_state.synthesis_part_name = None
    st.session_state.synthesis_body_name = None
    st.session_state.synthesis_intent = None
    st.session_state.synthesis_methods = []
    st.session_state.selected_synthesis_method = None
    st.session_state.synthesis_description = None
    st.session_state.synthesis_phase_d_ready = False

system: TrueCodingSystem = st.session_state.system
phase: str = st.session_state.phase

# サイドバー: 操作パネル
st.sidebar.title("操作パネル")
st.title(" 梅木卒論 実験用GUI")

# ========== Phase A: 初期入力 ==========
if phase == "A":
    st.sidebar.subheader("Phase A: 初期入力")
    
    # タブ構成: 通常 / 部分編集 / 合成
    input_tab1, input_tab2, input_tab3 = st.tabs(["📤 通常モード", "✂️ 部分編集モード", "🔨 合成モード"])
    
    # --- タブ1: 通常アップロード ---
    with input_tab1:
        st.subheader("画像をアップロード")
        up = st.file_uploader("粘土画像をアップロード", type=["png", "jpg", "jpeg"], key="normal_uploader")
        
        if up:
            img_path = _save_uploaded_image(up, prefix="initial")
            st.session_state.uploaded_image_path = img_path
            st.image(img_path, caption="アップロードされた画像", use_column_width=True)

    # --- タブ2: 部分編集モード ---
    with input_tab2:
        st.subheader("部分編集（インペインティング）")
        try:
            from streamlit_drawable_canvas import st_canvas
            part_up = st.file_uploader("編集対象の画像をアップロード", type=["png", "jpg", "jpeg"], key="partial_uploader")
            if part_up:
                base_path = _save_uploaded_image(part_up, prefix="partial_base")
                st.image(base_path, caption="編集対象の画像", use_column_width=True)

                st.markdown("#### 編集領域をマスクで指定")
                st.caption("赤色で編集したい領域を塗りつぶしてください（透明が編集対象）")
                base_img = _load_image(base_path)
                canvas_width = min(base_img.width, 800)
                canvas_height = int(base_img.height * (canvas_width / base_img.width))
                # Image オブジェクトをキャッシュ付きで読み込む（GC対策）
                canvas_image = _load_image(base_path)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=20,
                    stroke_color="#FF0000",
                    background_image=canvas_image,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="freedraw",
                    key="partial_canvas",
                )

                colp1, colp2, colp3 = st.columns([1,1,1])
                concept_partial = colp1.text_input("モチーフ (例: tank)", value="")
                target_name = colp2.text_input("部位名 (例: turret)", value="")
                partial_query = colp3.text_input("編集意図（クエリ）", value="")

                # 部分編集用の状態を初期化
                if "phase_a_partial_interpretations" not in st.session_state:
                    st.session_state.phase_a_partial_interpretations = []
                    st.session_state.phase_a_partial_mask_path = None
                    st.session_state.phase_a_partial_base_path = None
                    st.session_state.phase_a_partial_concept = None
                    st.session_state.phase_a_partial_target_name = None
                    st.session_state.phase_a_partial_query = None

                # ステップ1: 意図を解釈
                if st.button("意図を解釈（Interpret）", type="primary", key="phase_a_partial_interpret_btn"):
                    if not (concept_partial.strip() and target_name.strip() and partial_query.strip()):
                        st.warning("モチーフ・部位名・編集意図を入力してください")
                    elif canvas_result.image_data is None:
                        st.warning("マスクを描画してください")
                    else:
                        with st.spinner("マスク生成と意図解釈中..."):
                            # 透過マスク生成
                            mask_data = canvas_result.image_data
                            alpha_channel = mask_data[:, :, 3]
                            mask_bool = alpha_channel > 0
                            mask_uint8 = mask_bool.astype(np.uint8) * 255
                            temp_mask = Image.fromarray(mask_uint8, mode="L").resize(base_img.size, Image.Resampling.NEAREST)
                            final_mask = Image.new("RGBA", base_img.size, (0, 0, 0, 255))
                            mask_alpha = ImageOps.invert(temp_mask)
                            final_mask.putalpha(mask_alpha)
                            mask_path = Config.MASKS_PARTIAL_A_DIR / f"mask_{uuid.uuid4().hex}.png"
                            final_mask.save(mask_path)
                            
                            # セッション開始
                            try:
                                system.start_session(
                                    image_path=base_path,
                                    query=partial_query.strip(),
                                    concept=concept_partial.strip()
                                )
                            except Exception as e:
                                st.error(f"セッション開始に失敗: {e}")
                                st.stop()
                            
                            # クエリを解釈
                            try:
                                part_query = f"{target_name}: {partial_query}"
                                interps = system.interpret_query(query_override=part_query, use_image_context=False)
                                st.session_state.phase_a_partial_interpretations = [(i.id, i.text, i.reasoning) for i in interps]
                                st.session_state.phase_a_partial_mask_path = str(mask_path)
                                st.session_state.phase_a_partial_base_path = base_path
                                st.session_state.phase_a_partial_concept = concept_partial.strip()
                                st.session_state.phase_a_partial_target_name = target_name.strip()
                                st.session_state.phase_a_partial_query = partial_query.strip()
                                st.success("解釈を生成しました")
                                st.rerun()
                            except Exception as e:
                                st.error(f"解釈生成に失敗: {e}")
                
                # ステップ2: 解釈を選択して生成実行
                if st.session_state.phase_a_partial_interpretations:
                    st.markdown("---")
                    st.subheader("📋 解釈候補を選択")
                    options = [f"[{i}] {t}" for (i, t, r) in st.session_state.phase_a_partial_interpretations]
                    choice = st.radio("解釈を選んでください", options=options, index=0, key="phase_a_partial_interp_choice")
                    chosen_id = int(choice.split(']')[0][1:]) if choice else None
                    
                    with st.expander("根拠（Reasoning）"):
                        for (i, t, r) in st.session_state.phase_a_partial_interpretations:
                            st.markdown(f"- [{i}] {r}")
                    
                    if st.button("生成実行", type="primary", key="phase_a_partial_generate_btn"):
                        if chosen_id is None:
                            st.warning("解釈を選択してください")
                        else:
                            with st.spinner("Root作成と部分編集を実行中..."):
                                try:
                                    # 選択された解釈を適用
                                    system.select_interpretation(chosen_id)
                                    
                                    # 選択された解釈からモチーフを取得
                                    partial_motif = None
                                    if system.session.selected_interpretation:
                                        partial_motif = system.session.selected_interpretation.motif
                                    
                                    # Rootノード（グローバルベクトル）を画像分析から生成
                                    global_vec = system.vector_generator.generate_global_from_image(
                                        st.session_state.phase_a_partial_base_path,
                                        concept=st.session_state.phase_a_partial_concept
                                    )
                                    system.session.set_vector(global_vec)
                                    system.session.add_root_node(global_vec, system.session.constraints, note="root")
                                    
                                    # 部分ベクトル生成（選択された解釈から）
                                    if system.session.selected_interpretation:
                                        partial_vec = system.vector_generator.generate_vector_from_interpretation(
                                            system.session.selected_interpretation,
                                            constraints=[]
                                        )
                                    else:
                                        chosen_interpretation = next((i for i in st.session_state.phase_a_partial_interpretations if i[0] == chosen_id), None)
                                        part_text = chosen_interpretation[1] if chosen_interpretation else f"{st.session_state.phase_a_partial_target_name}: {st.session_state.phase_a_partial_query}"
                                        partial_vec = system.vector_generator.generate_from_text(
                                            part_text,
                                            concept=st.session_state.phase_a_partial_concept,
                                            max_attrs=8
                                        )
                                    
                                    # インペインティングで部分編集
                                    result_path = system.image_generator.generate_part_from_vector(
                                        base_image_path=st.session_state.phase_a_partial_base_path,
                                        mask_path=st.session_state.phase_a_partial_mask_path,
                                        partial_vector=partial_vec,
                                        concept=st.session_state.phase_a_partial_concept,
                                        target_part_name=st.session_state.phase_a_partial_target_name,
                                        partial_motif=partial_motif
                                    )
                                    
                                    # 子ノード作成
                                    from models.session import GeneratedImage
                                    system.session.add_child_node(
                                        vector=global_vec,
                                        constraints=system.session.constraints,
                                        note=f"partial_edit:{st.session_state.phase_a_partial_target_name}",
                                        partial_vector=partial_vec,
                                        mask_image_path=st.session_state.phase_a_partial_mask_path,
                                        target_part_name=st.session_state.phase_a_partial_target_name,
                                        partial_motif=partial_motif
                                    )
                                    system.session.update_current_node_image(result_path)
                                    system.session.add_generated_image(
                                        GeneratedImage(
                                            image_path=result_path,
                                            prompt="partial_edit",
                                            vector=partial_vec,
                                            constraints=system.session.constraints
                                        )
                                    )
                                    
                                    # 状態をリセット
                                    st.session_state.phase_a_partial_interpretations = []
                                    st.session_state.phase_a_partial_mask_path = None
                                    st.session_state.phase_a_partial_base_path = None
                                    st.session_state.phase_a_partial_concept = None
                                    st.session_state.phase_a_partial_target_name = None
                                    st.session_state.phase_a_partial_query = None
                                    
                                    st.success("部分編集を完了しました！")
                                    st.image(result_path, caption="部分編集結果", use_column_width=True)
                                    st.session_state.phase = "D"
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"部分編集に失敗: {e}")
                    
                    if st.button("解釈をやり直す", key="phase_a_partial_reset_btn"):
                        st.session_state.phase_a_partial_interpretations = []
                        st.rerun()
        except ImportError:
            st.error("streamlit-drawable-canvas がインストールされていません。")
            st.code("pip install streamlit-drawable-canvas")
    
    # --- タブ3: 合成モード ---
    with input_tab3:
        st.subheader("粘土パーツを合成")
        
        # 合成モード用の状態初期化
        if "synthesis_base_path" not in st.session_state:
            st.session_state.synthesis_base_path = None
            st.session_state.synthesis_mask_path = None
            st.session_state.synthesis_part_name = None
            st.session_state.synthesis_body_name = None
            st.session_state.synthesis_intent = None
            st.session_state.synthesis_methods = []
            st.session_state.selected_synthesis_method = None
            st.session_state.synthesis_result_path = None
            st.session_state.synthesis_description = None  # 追加: 画像説明をキャッシュ
            st.session_state.synthesis_phase_d_ready = False  # 追加: フェーズD準備完了フラグ
        
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # ユーザーへの配置指示
            st.info("💡 撮影のルール: 合成したい「パーツ」を左側に、「本体」を右側に並べて撮影してください。")

            # ステップ1: 画像とマスク、指示を入力
            synthesis_up = st.file_uploader(
                "合成元画像をアップロード（パーツと本体を写した画像）",
                type=["png", "jpg", "jpeg"],
                key="synthesis_uploader"
            )
            
            if synthesis_up:
                synthesis_base_path = _save_uploaded_image(synthesis_up, prefix="synthesis_base")
                st.session_state.synthesis_base_path = synthesis_base_path
                st.image(synthesis_base_path, caption="合成元画像", use_column_width=True)
            
            # セッション状態から画像パスを取得（一度アップロードされていれば再利用）
            if hasattr(st.session_state, 'synthesis_base_path') and st.session_state.synthesis_base_path:
                synthesis_base_path = st.session_state.synthesis_base_path
                
                # 入力フォームの構造化
                col_names1, col_names2 = st.columns(2)
                part_name = col_names1.text_input("左側のパーツ名 (例: cannon)", key="syn_part_name")
                body_name = col_names2.text_input("右側の本体名 (例: tank body)", key="syn_body_name")

                st.markdown("#### 合成場所をマスクで指定")
                st.caption("赤色で合成したい領域（接合部）を塗りつぶしてください")
                
                # Canvas for drawing mask
                base_img = _load_image(synthesis_base_path)
                canvas_width = min(base_img.width, 800)
                canvas_height = int(base_img.height * (canvas_width / base_img.width))
                # Image オブジェクトをキャッシュ付きで読み込む（GC対策）
                canvas_image = _load_image(synthesis_base_path)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=20,
                    stroke_color="#FF0000",
                    background_image=canvas_image,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="freedraw",
                    key="canvas_synthesis",
                )
                
                # ユーザー意図（任意）
                synthesis_intent = st.text_input(
                    "具体的な指示・意図（任意）",
                    placeholder="例: 滑らかに繋げて、本体の上に乗せて",
                    key="synthesis_intent_input"
                )

                # synthesis_instruction = st.text_input(
                #     "合成指示（例: 左のパーツを右の本体にくっつけて）",
                #     value="",
                #     key="synthesis_instruction_input"
                # )
                
                # ステップ2: 解釈フェーズ
                if st.button("接合方法を解釈（Interpret）", type="primary", key="synthesis_interpret_btn"):
                    if canvas_result.image_data is not None and part_name and body_name:
                        with st.spinner("接合方法を解釈中..."):
                            # マスク画像を透過形式で作成
                            mask_data = canvas_result.image_data
                            alpha_channel = mask_data[:, :, 3]
                            mask_bool = alpha_channel > 0
                            mask_uint8 = mask_bool.astype(np.uint8) * 255
                            temp_mask = Image.fromarray(mask_uint8, mode="L")
                            temp_mask = temp_mask.resize(base_img.size, Image.Resampling.NEAREST)
                            
                            # 透過マスクを作成
                            final_mask = Image.new("RGBA", base_img.size, (0, 0, 0, 255))
                            mask_alpha = ImageOps.invert(temp_mask)
                            final_mask.putalpha(mask_alpha)
                            
                            # マスクを保存
                            mask_path = Config.MASKS_SYNTHESIS_DIR / f"mask_{uuid.uuid4().hex}.png"
                            final_mask.save(mask_path)

                            # 状態保存
                            st.session_state.synthesis_mask_path = str(mask_path)
                            st.session_state.synthesis_part_name = part_name
                            st.session_state.synthesis_body_name = body_name
                            st.session_state.synthesis_intent = synthesis_intent
                            
                            # 接合方法を解釈（引数を更新）
                            try:
                                synthesis_methods = system.query_interpreter.interpret_synthesis_method(
                                    part_name=part_name,
                                    body_name=body_name,
                                    user_intent=synthesis_intent,
                                    image_path=synthesis_base_path
                                )
                                st.session_state.synthesis_methods = synthesis_methods
                                st.success("接合方法を解釈しました！")
                            except Exception as e:
                                st.error(f"解釈に失敗: {e}")
                    else:
                        st.warning("パーツ名、本体名、およびマスクを入力してください")
                
                # ステップ3: 接合方法を表示・選択
                if st.session_state.synthesis_methods:
                    st.markdown("#### 接合方法の提案")
                    
                    for idx, method in enumerate(st.session_state.synthesis_methods):
                        col1, col2 = st.columns([1, 20])
                        with col1:
                            st.write("")
                        with col2:
                            st.markdown(f"**案{idx+1}:** {method.text}")
                            if method.reasoning:
                                st.caption(f"理由: {method.reasoning}")
                        
                        # 選択ボタン
                        if st.button(f"この方法を選択", key=f"select_synthesis_{idx}"):
                            st.session_state.selected_synthesis_method = method
                            st.session_state.synthesis_result_path = None  # リセット
                            st.success(f"案{idx+1}を選択しました")
                
                # ステップ4: 生成フェーズ
                if st.session_state.selected_synthesis_method:
                    st.markdown("#### 選択された接合方法")
                    st.info(st.session_state.selected_synthesis_method.text)
                    
                    if st.button("合成を実行", type="primary", key="synthesis_generate_btn"):
                        with st.spinner("粘土パーツを合成中..."):
                            try:
                                # 選択された接合方法でプロンプトを生成
                                method_text = st.session_state.selected_synthesis_method.text
                                
                                # result_path = system.image_generator.generate_synthesis(
                                #     st.session_state.synthesis_base_path,
                                #     st.session_state.synthesis_mask_path,
                                #     st.session_state.synthesis_instruction,
                                #     synthesis_method=method_text
                                # )
                                # 新しい引数で呼び出し
                                result_path = system.image_generator.generate_synthesis(
                                    base_image_path=st.session_state.synthesis_base_path,
                                    mask_path=st.session_state.synthesis_mask_path,
                                    part_name=st.session_state.synthesis_part_name, # 追加
                                    body_name=st.session_state.synthesis_body_name, # 追加
                                    synthesis_method=method_text,
                                    user_intent=st.session_state.synthesis_intent     # 追加
                                )
                                st.session_state.synthesis_result_path = result_path
                                st.success("合成が完了しました！")
                                st.image(result_path, caption="合成結果", use_column_width=True)
                                
                                # 生成画像を解析して説明を取得（キャッシュ）
                                try:
                                    desc = system.image_generator.extract_description_from_image(result_path)
                                    st.session_state.synthesis_description = desc
                                    st.session_state.synthesis_phase_d_ready = True
                                except Exception as e:
                                    st.error(f"画像説明生成に失敗: {e}")
                            except Exception as e:
                                st.error(f"合成に失敗: {e}")
        
        except ImportError:
            st.error("streamlit-drawable-canvas がインストールされていません。")
            st.code("pip install streamlit-drawable-canvas")
    
    # --- モチーフ（Concept）とクエリ入力 ---
    st.markdown("---")
    st.subheader("モチーフと意図を入力")
    
    # 合成モードで準備完了している場合
    if st.session_state.synthesis_phase_d_ready:
        st.markdown("#### 合成結果からフェーズDへ")
        st.info(f"画像の説明: {st.session_state.synthesis_description}")
        
        concept_for_synthesis = st.session_state.synthesis_body_name
        
        col_syn_start1, col_syn_start2 = st.columns([2, 1])
        with col_syn_start1:
            st.write(f"**コンセプト:** {concept_for_synthesis}")  # 入力欄の代わりに表示
        with col_syn_start2:
            if st.button("フェーズDを開始", type="primary", key="start_phase_d_synthesis"):
                with st.spinner("セッションを初期化中..."):
                    try:
                        # セッションを初期化（必須）
                        system.start_session(
                            image_path=st.session_state.synthesis_result_path,
                            query=st.session_state.synthesis_intent.strip() if st.session_state.synthesis_intent else "合成モード",
                            concept=concept_for_synthesis.strip()
                        )
                        
                        # 全体ベクトルを生成
                        global_vector = system.vector_generator.generate_from_text(
                            concept_for_synthesis,
                            concept=concept_for_synthesis,
                            max_attrs=15
                        )
                        
                        # セッションの根ノードを作成
                        system.session.add_root_node(
                            vector=global_vector,
                            constraints=[],
                            note="synthesis_root",
                            motif=concept_for_synthesis.strip()
                        )
                        
                        # 根ノードに生成画像を関連付け
                        system.session.update_current_node_image(st.session_state.synthesis_result_path)
                        from models.session import GeneratedImage
                        system.session.add_generated_image(
                            GeneratedImage(
                                image_path=st.session_state.synthesis_result_path,
                                prompt="synthesis_root",
                                vector=global_vector,
                                constraints=[]
                            )
                        )
                        
                        # フェーズD初期化
                        st.session_state.phase = "D"
                        st.session_state.uploaded_image_path = st.session_state.synthesis_result_path
                        st.session_state.concept = concept_for_synthesis.strip()
                        
                        st.success(f"フェーズDを開始しました（コンセプト: {concept_for_synthesis}）")
                        st.session_state.synthesis_phase_d_ready = False  # リセット
                        st.rerun()
                    except Exception as e:
                        st.error(f"フェーズD開始に失敗: {e}")
        st.markdown("---")
    
    final_image_path = st.session_state.uploaded_image_path
    
    if final_image_path:
        st.image(final_image_path, caption="使用する画像", width=300)
        
        col_concept, col_query = st.columns(2)
        concept = col_concept.text_input(
            "これ（モチーフ）は何ですか？（必須）",
            value="",
            placeholder="Tank, Vase, Chair ...",
            help="英語の単数形を推奨します (例: Tank, Vase, Chair)"
        )
        query = col_query.text_input(
            "デザインの意図（クエリ）",
            value="",
            placeholder="角張った未来的なフォルムにして"
        )
        
        if st.button("セッション開始", type="primary", key="start_session_btn"):
            if concept.strip() and query.strip():
                try:
                    system.start_session(
                        image_path=final_image_path,
                        query=query.strip(),
                        concept=concept.strip()
                    )
                    st.session_state.phase = "B"
                    st.session_state.interpretations = []
                    st.success("セッションを開始しました")
                    st.rerun()
                except Exception as e:
                    st.error(f"開始に失敗: {e}")
            else:
                st.warning("モチーフとクエリを入力してください")
    else:
        if not st.session_state.synthesis_phase_d_ready:
            st.info("画像をアップロードしてください")

# ========== Phase B: 解釈選択 ==========
elif phase == "B":
    st.header("Phase B: 解釈選択")

    # 初回のみ解釈を生成（再描画で多重実行を避ける）
    if not st.session_state.interpretations:
        try:
            interps = system.interpret_query(use_image_context=True)
            # 保持用に簡素化
            st.session_state.interpretations = [(i.id, i.text, i.reasoning) for i in interps]
        except Exception as e:
            st.error(f"解釈生成に失敗: {e}")

    # メインエリア: 解釈の提示
    if st.session_state.interpretations:
        st.subheader("候補の解釈")
        options = [f"[{i}] {t}" for (i, t, r) in st.session_state.interpretations]
        choice = st.radio("解釈を選んでください", options=options, index=0)
        chosen_id = int(choice.split(']')[0][1:]) if choice else None
        with st.expander("根拠（Reasoning）"):
            for (i, t, r) in st.session_state.interpretations:
                st.markdown(f"- [{i}] {r}")

        # 決定ボタン
        if st.button("この解釈で進む", type="primary"):
            try:
                with st.spinner("解釈を適用中..."):
                    system.select_interpretation(chosen_id)
                with st.spinner("特徴ベクトルを生成中..."):
                    system.generate_vector()
                with st.spinner("画像を生成中...（数十秒かかる場合があります）"):
                    system.generate_image()
                st.success("画像生成が完了しました！")
                st.session_state.phase = "D"
                st.rerun()
            except Exception as e:
                st.error(f"進行に失敗: {e}")
    else:
        st.warning("解釈候補がありません。再生成してください。")

    # サイドバー: 再生成フロー
    st.sidebar.subheader("解釈を再生成")
    refined_text = st.sidebar.text_input("クエリを再入力（詳細化）")
    add_img = st.sidebar.file_uploader("補足用の画像（任意）", type=["png", "jpg", "jpeg"], key="refine_uploader")
    if st.sidebar.button("解釈を再生成"):
        try:
            if refined_text.strip():
                system.refine_query(refined_text.strip())
            add_path = _save_uploaded_image(add_img, prefix="additional") if add_img else None
            interps = system.interpret_query(additional_image=add_path, use_image_context=True)
            st.session_state.interpretations = [(i.id, i.text, i.reasoning) for i in interps]
            st.success("解釈を再生成しました")
        except Exception as e:
            st.sidebar.error(f"再生成に失敗: {e}")

    # 属性一覧テーブル（日本語のみ）
    st.markdown("---")
    st.markdown("### 属性一覧")
    for gname, items in ATTR_SPACE.groups.items():
        st.subheader(f"🏷️ {gname}")
        attr_names = list(items.values())
        cols_data = [[] for _ in range(3)]
        for idx, name in enumerate(attr_names):
            cols_data[idx % 3].append(name)
        max_len = max(len(col) for col in cols_data)
        for col in cols_data:
            while len(col) < max_len:
                col.append("")
        table_data = {f"属性{i+1}": cols_data[i] for i in range(3)}
        df_attrs = pd.DataFrame(table_data)
        st.dataframe(df_attrs, use_container_width=True, hide_index=True)

# ========== Phase D: 生成と探索（メインループ） ==========
elif phase == "D":
    st.header("Phase D: 生成と探索")

    # 上部: 画像2カラム
    import os
    col1, col2 = st.columns(2)
    if system.session and system.session.initial_image_path and os.path.exists(system.session.initial_image_path):
        col1.subheader("元画像")
        col1.image(system.session.initial_image_path, use_column_width=True)
    latest = system.session.get_latest_image() if system.session else None
    if latest and latest.image_path and os.path.exists(latest.image_path):
        col2.subheader("生成画像")
        col2.image(latest.image_path, use_column_width=True)
        if st.button("この画像を分析", key="analyze_btn"):
            try:
                st.session_state.analysis_text = system.analyze_current_image()
            except Exception as e:
                st.error(f"分析に失敗: {e}")
    if st.session_state.analysis_text:
        st.markdown("### 画像分析結果")
        st.write(st.session_state.analysis_text)

    # サイドバー: セッション管理
    st.sidebar.subheader("セッション管理")
    if system.session:
        summary = system.get_session_summary()
        st.sidebar.download_button(
            label="セッション要約をダウンロード (JSON)",
            data=json.dumps(summary, ensure_ascii=False, indent=2),
            file_name=f"session_{summary.get('session_id','unknown')}.json",
            use_container_width=True,
        )
        # 最新画像のダウンロード
        latest = system.session.get_latest_image()
        if latest and latest.image_path:
            try:
                ext = Path(latest.image_path).suffix.lower()
                mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                with open(latest.image_path, "rb") as f:
                    st.sidebar.download_button(
                        label="最新画像をダウンロード",
                        data=f.read(),
                        file_name=f"latest{ext or '.jpg'}",
                        mime=mime,
                        use_container_width=True,
                    )
            except Exception as e:
                st.sidebar.error(f"画像ダウンロードの準備に失敗: {e}")

    if st.sidebar.button("セッション終了", type="secondary"):
        # 最新画像をoutputsに保存コピー
        try:
            latest = system.session.get_latest_image() if system.session else None
            if latest and latest.image_path:
                out_dir = Config.OUTPUTS_DIR
                out_dir.mkdir(parents=True, exist_ok=True)
                ext = Path(latest.image_path).suffix.lower() or ".jpg"
                fname = f"final_{system.session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                dest = out_dir / fname
                shutil.copy2(latest.image_path, dest)
                st.sidebar.success(f"最新画像を保存: {dest}")
        except Exception as e:
            st.sidebar.error(f"最新画像の保存に失敗: {e}")

        # セッション状態を初期化してPhase Aへ戻す
        st.session_state.system = TrueCodingSystem()
        st.session_state.phase = "A"
        st.session_state.interpretations = []
        st.session_state.selected_interpretation_id = None
        st.session_state.analysis_text = ""
        st.session_state.search_results = []
        st.success("セッションを終了しました。新しいセッションを開始できます。")
        st.rerun()

    # サイドバー: 全体調整（制約）
    st.sidebar.subheader("調整タブ")
    st.sidebar.markdown("#### 制約を追加（グローバルベクトル）")
    
    # セレクトボックス用オプション（グループ順を保持）
    all_attrs: List[Tuple[str, str]] = []  # (display, key)
    for gname, items in ATTR_SPACE.groups.items():
        for k, v in items.items():
            key = f"{gname}:{k}"
            display = f"{v} ({key})"
            all_attrs.append((display, key))
    display_labels = [d for d, k in all_attrs]
    selected_display = st.sidebar.selectbox("属性キー", options=display_labels)
    selected_key = dict(all_attrs)[selected_display]

    ctype_map = {
        "≥ 以上": "greater_than",
        "≤ 以下": "less_than",
        "= 等しい": "equal",
        "範囲指定": "range",
    }
    ctype_label = st.sidebar.radio("制約タイプ", options=list(ctype_map.keys()), horizontal=True)
    ctype = ctype_map[ctype_label]

    val = None
    min_val = None
    max_val = None
    if ctype == "range":
        min_val, max_val = st.sidebar.slider("値の範囲", min_value=0.0, max_value=1.0, value=(0.4, 0.6), step=0.05)
    else:
        val = st.sidebar.slider("値", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    desc = st.sidebar.text_input("説明（任意）", value="")
    if st.sidebar.button("制約を追加して再生成", type="primary"):
        try:
            system.add_constraint(
                attribute_key=selected_key,
                constraint_type=ctype,
                value=float(val) if val is not None else None,
                min_value=float(min_val) if min_val is not None else None,
                max_value=float(max_val) if max_val is not None else None,
                description=desc,
            )
            
            # 現在のノードが部分編集ノードかどうかをチェック
            current_node = None
            if system.session and system.session.current_node_id is not None:
                current_node = next((n for n in system.session.exploration_nodes if n.node_id == system.session.current_node_id), None)
            
            if current_node and current_node.partial_vector and current_node.mask_image_path:
                # 部分編集ノードの場合：部分ベクトルに制約を適用して再編集
                from models.attribute_space import AttributeVector
                import os
                
                # 親ノードの画像をベースとして取得
                base_path = None
                if current_node.parent_id is not None:
                    parent_node = next((n for n in system.session.exploration_nodes if n.node_id == current_node.parent_id), None)
                    if parent_node and parent_node.generated_image_path and os.path.exists(parent_node.generated_image_path):
                        base_path = parent_node.generated_image_path
                
                # フォールバック1: 最新の生成画像
                if not base_path:
                    latest = system.session.get_latest_image()
                    if latest and latest.image_path and os.path.exists(latest.image_path):
                        base_path = latest.image_path
                
                # フォールバック2: 初期画像
                if not base_path and system.session.initial_image_path and os.path.exists(system.session.initial_image_path):
                    base_path = system.session.initial_image_path
                
                if not base_path or not os.path.exists(base_path):
                    st.sidebar.error(f"ベース画像が見つかりません（親: {current_node.parent_id}, パス: {base_path}）")
                else:
                    # 部分ベクトルのコピーを作成し、制約に従って調整
                    partial_weights = dict(current_node.partial_vector.weights)
                    
                    # 最新の制約を適用（シンプルな実装）
                    for constraint in system.session.constraints:
                        attr_key = constraint.attribute_key
                        if constraint.constraint_type == "greater_than" and constraint.value is not None:
                            if attr_key in partial_weights:
                                partial_weights[attr_key] = max(partial_weights[attr_key], constraint.value)
                            else:
                                partial_weights[attr_key] = constraint.value
                        elif constraint.constraint_type == "less_than" and constraint.value is not None:
                            if attr_key in partial_weights:
                                partial_weights[attr_key] = min(partial_weights[attr_key], constraint.value)
                            else:
                                partial_weights[attr_key] = constraint.value
                        elif constraint.constraint_type == "equal" and constraint.value is not None:
                            partial_weights[attr_key] = constraint.value
                        elif constraint.constraint_type == "range" and constraint.min_value is not None and constraint.max_value is not None:
                            if attr_key in partial_weights:
                                partial_weights[attr_key] = max(constraint.min_value, min(constraint.max_value, partial_weights[attr_key]))
                            else:
                                partial_weights[attr_key] = (constraint.min_value + constraint.max_value) / 2
                    
                    updated_partial_vec = AttributeVector(weights=partial_weights)
                    
                    with st.spinner("制約を適用して部分編集を再生成中..."):
                        result_path = system.image_generator.generate_part_from_vector(
                            base_image_path=base_path,
                            mask_path=str(current_node.mask_image_path),
                            partial_vector=updated_partial_vec,
                            concept=system.session.concept,
                            target_part_name=current_node.target_part_name,
                            partial_motif=current_node.partial_motif
                        )
                        
                        # 子ノードを作成
                        from models.session import GeneratedImage
                        system.session.add_child_node(
                            vector=current_node.vector,
                            constraints=system.session.constraints,
                            note=f"constraint_partial:{current_node.target_part_name}",
                            partial_vector=updated_partial_vec,
                            mask_image_path=current_node.mask_image_path,
                            target_part_name=current_node.target_part_name,
                            partial_motif=current_node.partial_motif
                        )
                        system.session.update_current_node_image(result_path)
                        system.session.add_generated_image(
                            GeneratedImage(
                                image_path=result_path,
                                prompt="constraint_partial",
                                vector=updated_partial_vec,
                                constraints=system.session.constraints
                            )
                        )
                        st.success("制約を適用して部分編集を再生成しました")
            else:
                # 通常の全体画像として再生成
                with st.spinner("制約を適用して全体画像を再生成中..."):
                    system.generate_image()
                    st.success("画像を再生成しました")
            
            st.rerun()
        except Exception as e:
            st.error(f"制約の適用に失敗: {e}")
            import traceback
            st.error(traceback.format_exc())

    # サイドバー: 微調整セクション
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 制約ベース微調整（部分編集ノード）")
    try:
        # 現在ノードを取得
        selected_node_detail_for_tuning = None
        if system.session and system.session.current_node_id is not None:
            selected_node_detail_for_tuning = next((n for n in system.session.exploration_nodes if n.node_id == system.session.current_node_id), None)
        if selected_node_detail_for_tuning and getattr(selected_node_detail_for_tuning, "partial_vector", None):
            pv = selected_node_detail_for_tuning.partial_vector
            from models.attribute_space import AttributeVector
            from models.constraints import Constraint, ConstraintType
            
            # 現在の属性値を表示（読み取り専用）
            st.sidebar.markdown("**現在の部分ベクトル:**")
            top_items = sorted(pv.weights.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            for attr_key, weight in top_items:
                name = ATTR_SPACE.get_attribute_name(attr_key) or attr_key
                st.sidebar.text(f"{name}: {weight:.3f}")
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("**制約を設定:**")
            
            # 属性選択（上位属性+全属性から選択可能）
            all_attrs = ATTR_SPACE.all_attributes  # これはList[str]
            top_attr_keys = [item[0] for item in top_items]
            # 上位属性を最初に、残りをアルファベット順で
            attr_options = top_attr_keys + [k for k in sorted(all_attrs) if k not in top_attr_keys]
            attr_names = [f"{ATTR_SPACE.get_attribute_name(k) or k} ({k})" for k in attr_options]
            
            selected_attr_idx = st.sidebar.selectbox(
                "調整したい属性",
                range(len(attr_options)),
                format_func=lambda i: attr_names[i],
                key=f"constraint_attr_{selected_node_detail_for_tuning.node_id}"
            )
            selected_attr_key = attr_options[selected_attr_idx]
            
            # 不等号選択
            operator = st.sidebar.radio(
                "制約タイプ",
                [">=", "<=", "=="],
                index=0,
                key=f"constraint_op_{selected_node_detail_for_tuning.node_id}",
                horizontal=True
            )
            
            # 値設定
            current_value = pv.weights.get(selected_attr_key, 0.0)
            target_value = st.sidebar.slider(
                "目標値",
                min_value=0.0,
                max_value=1.0,
                value=float(max(0.0, min(1.0, current_value))),
                step=0.05,
                key=f"constraint_val_{selected_node_detail_for_tuning.node_id}"
            )
            
            # 制約説明
            attr_name = ATTR_SPACE.get_attribute_name(selected_attr_key) or selected_attr_key
            st.sidebar.caption(f"制約: {attr_name} {operator} {target_value}")

            if st.sidebar.button("制約を適用して再編集", key=f"constraint_apply_{selected_node_detail_for_tuning.node_id}"):
                if not selected_node_detail_for_tuning.mask_image_path or not selected_node_detail_for_tuning.target_part_name:
                    st.sidebar.warning("このノードにはマスクまたは部位名情報がありません")
                else:
                    base_path = None
                    latest = system.session.get_latest_image() if system.session else None
                    if latest and latest.image_path:
                        base_path = latest.image_path
                    elif system.session and system.session.initial_image_path:
                        base_path = system.session.initial_image_path
                    if not base_path:
                        st.sidebar.warning("ベース画像が見つかりません")
                    else:
                        # 制約を作成
                        constraint_type = {
                            ">=": ConstraintType.GREATER_THAN,
                            "<=": ConstraintType.LESS_THAN, 
                            "==": ConstraintType.EQUAL
                        }[operator]
                        
                        constraint = Constraint(
                            attribute=selected_attr_key,
                            constraint_type=constraint_type,
                            value=target_value,
                            description=f"{attr_name} {operator} {target_value}"
                        )
                        
                        # 制約を適用して部分ベクトルを更新
                        updated_pv = system.vector_generator.update_vector_with_constraint(
                            current_vector=pv,
                            constraint=constraint,
                            interpretation=None
                        )
                        
                        # 画像生成
                        result_path = system.image_generator.generate_part_from_vector(
                            base_image_path=base_path,
                            mask_path=str(selected_node_detail_for_tuning.mask_image_path),
                            partial_vector=updated_pv,
                            concept=system.session.concept,
                            target_part_name=selected_node_detail_for_tuning.target_part_name,
                            partial_motif=selected_node_detail_for_tuning.partial_motif
                        )
                        
                        # 子ノード作成
                        from models.session import GeneratedImage
                        system.session.add_child_node(
                            vector=selected_node_detail_for_tuning.vector,
                            constraints=system.session.constraints + [constraint],
                            note=f"constraint_tune:{selected_node_detail_for_tuning.target_part_name}",
                            partial_vector=updated_pv,
                            mask_image_path=selected_node_detail_for_tuning.mask_image_path,
                            target_part_name=selected_node_detail_for_tuning.target_part_name,
                            partial_motif=selected_node_detail_for_tuning.partial_motif
                        )
                        system.session.update_current_node_image(result_path)
                        system.session.add_generated_image(GeneratedImage(image_path=result_path, prompt="constraint_tune", vector=updated_pv, constraints=system.session.constraints + [constraint]))
                        
                        st.sidebar.success(f"制約「{attr_name} {operator} {target_value}」を適用して再編集しました")
                        st.rerun()
        else:
            st.sidebar.info("部分編集ノードを選択すると制約ベース微調整UIが表示されます")
    except Exception as e:
        st.sidebar.error(f"微調整に失敗: {e}")

    # メインエリア: 新規部分編集セクション（2段階フロー）
    st.markdown("---")
    st.subheader("✂️ 新規部分編集")
    
    # 部分編集用の状態を初期化（初回のみ）
    if "partial_interpretations" not in st.session_state:
        st.session_state.partial_interpretations = []
        st.session_state.selected_partial_interpretation_id = None
        st.session_state.partial_edit_mask_path = None
        st.session_state.partial_edit_target_name = None
        st.session_state.partial_edit_base_path = None
        st.session_state.partial_edit_query = None
    
    try:
        from streamlit_drawable_canvas import st_canvas
        import os
        
        # ベースは現在ノードの画像を使用
        base_path = None
        if system.session and system.session.current_node_id is not None:
            current_node = next((n for n in system.session.exploration_nodes if n.node_id == system.session.current_node_id), None)
            if current_node and current_node.generated_image_path and os.path.exists(current_node.generated_image_path):
                base_path = current_node.generated_image_path
        # フォールバック：現在ノードに画像がない場合は最新画像または初期画像
        if not base_path:
            latest = system.session.get_latest_image() if system.session else None
            if latest and latest.image_path and os.path.exists(latest.image_path):
                base_path = latest.image_path
            elif system.session and system.session.initial_image_path and os.path.exists(system.session.initial_image_path):
                base_path = system.session.initial_image_path
        
        if base_path:
            # ステップ1: マスク描画と意図入力
            col_img, col_form = st.columns([1, 1])
            with col_img:
                st.caption("編集対象の画像")
                st.image(base_path, use_column_width=True)
            
            with col_form:
                st.caption("マスク描画（赤色で編集領域を指定）")
                base_img = _load_image(base_path)
                canvas_width = 300
                canvas_height = int(base_img.height * (canvas_width / base_img.width))
                # Image オブジェクトをキャッシュ付きで読み込む（GC対策）
                canvas_image = _load_image(base_path)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=15,
                    stroke_color="#FF0000",
                    background_image=canvas_image,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="freedraw",
                    key="partial_canvas_main",
                )
            
            col_input1, col_input2 = st.columns(2)
            target_name = col_input1.text_input("部位名（例: turret）", value="", key="partial_target_name")
            partial_query = col_input2.text_input("編集意図（例: 砲塔を追加）", value="", key="partial_query_input")
            
            # ステップ2: 意図を解釈
            if st.button("意図を解釈（Interpret）", type="primary", key="partial_interpret_btn"):
                if not (target_name.strip() and partial_query.strip()):
                    st.warning("部位名と編集意図を入力してください")
                elif canvas_result.image_data is None:
                    st.warning("マスクを描画してください")
                else:
                    with st.spinner("編集意図を解釈中..."):
                        # マスク生成して保存
                        mask_data = canvas_result.image_data
                        alpha_channel = mask_data[:, :, 3]
                        mask_bool = alpha_channel > 0
                        mask_uint8 = mask_bool.astype(np.uint8) * 255
                        temp_mask = Image.fromarray(mask_uint8, mode="L").resize(base_img.size, Image.Resampling.NEAREST)
                        final_mask = Image.new("RGBA", base_img.size, (0, 0, 0, 255))
                        mask_alpha = ImageOps.invert(temp_mask)
                        final_mask.putalpha(mask_alpha)
                        mask_path = Config.MASKS_PARTIAL_D_DIR / f"mask_{uuid.uuid4().hex}.png"
                        final_mask.save(mask_path)
                        
                        # セッション状態に保存
                        st.session_state.partial_edit_mask_path = str(mask_path)
                        st.session_state.partial_edit_target_name = target_name.strip()
                        st.session_state.partial_edit_base_path = base_path
                        st.session_state.partial_edit_query = partial_query.strip()
                        
                        # クエリを解釈（部位名を含めたクエリで解釈）
                        try:
                            part_query = f"{target_name}: {partial_query}"
                            interps = system.interpret_query(query_override=part_query, use_image_context=False)
                            st.session_state.partial_interpretations = [(i.id, i.text, i.reasoning) for i in interps]
                            st.success("解釈を生成しました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"解釈生成に失敗: {e}")
            
            # ステップ3: 解釈を選択して生成実行
            if st.session_state.partial_interpretations:
                st.markdown("---")
                st.subheader("📋 解釈候補を選択")
                options = [f"[{i}] {t}" for (i, t, r) in st.session_state.partial_interpretations]
                choice = st.radio("解釈を選んでください", options=options, index=0, key="partial_interp_choice")
                chosen_id = int(choice.split(']')[0][1:]) if choice else None
                
                with st.expander("根拠（Reasoning）"):
                    for (i, t, r) in st.session_state.partial_interpretations:
                        st.markdown(f"- [{i}] {r}")
                
                if st.button("生成実行", type="primary", key="partial_generate_btn"):
                    if chosen_id is None:
                        st.warning("解釈を選択してください")
                    else:
                        with st.spinner("部分編集を実行中..."):
                            try:
                                # 選択された解釈を適用
                                system.select_interpretation(chosen_id)
                                
                                # 選択された解釈からモチーフを取得
                                partial_motif = None
                                if system.session.selected_interpretation:
                                    partial_motif = system.session.selected_interpretation.motif
                                
                                # 部分ベクトル生成（選択された解釈から）
                                if system.session.selected_interpretation:
                                    partial_vec = system.vector_generator.generate_vector_from_interpretation(
                                        system.session.selected_interpretation,
                                        constraints=[]
                                    )
                                else:
                                    # フォールバック
                                    chosen_interpretation = next((i for i in st.session_state.partial_interpretations if i[0] == chosen_id), None)
                                    part_text = chosen_interpretation[1] if chosen_interpretation else f"{st.session_state.partial_edit_target_name}: {st.session_state.partial_edit_query}"
                                    partial_vec = system.vector_generator.generate_from_text(
                                        part_text, 
                                        concept=system.session.concept, 
                                        max_attrs=8
                                    )
                                
                                # 実行
                                result_path = system.image_generator.generate_part_from_vector(
                                    base_image_path=st.session_state.partial_edit_base_path,
                                    mask_path=st.session_state.partial_edit_mask_path,
                                    partial_vector=partial_vec,
                                    concept=system.session.concept,
                                    target_part_name=st.session_state.partial_edit_target_name,
                                    partial_motif=partial_motif
                                )
                                
                                # 親ノードのグローバルベクトルを取得
                                parent_vector = system.session.current_vector
                                if system.session.current_node_id is not None:
                                    parent_node = next((n for n in system.session.exploration_nodes if n.node_id == system.session.current_node_id), None)
                                    if parent_node and parent_node.vector:
                                        parent_vector = parent_node.vector
                                
                                # 子ノード作成
                                from models.session import GeneratedImage
                                system.session.add_child_node(
                                    vector=parent_vector or partial_vec,
                                    constraints=system.session.constraints,
                                    note=f"partial_edit:{st.session_state.partial_edit_target_name}",
                                    partial_vector=partial_vec,
                                    mask_image_path=st.session_state.partial_edit_mask_path,
                                    target_part_name=st.session_state.partial_edit_target_name,
                                    partial_motif=partial_motif
                                )
                                system.session.update_current_node_image(result_path)
                                system.session.add_generated_image(
                                    GeneratedImage(
                                        image_path=result_path, 
                                        prompt="partial_edit", 
                                        vector=partial_vec, 
                                        constraints=system.session.constraints
                                    )
                                )
                                
                                # 状態をリセット
                                st.session_state.partial_interpretations = []
                                st.session_state.selected_partial_interpretation_id = None
                                st.session_state.partial_edit_mask_path = None
                                st.session_state.partial_edit_target_name = None
                                st.session_state.partial_edit_base_path = None
                                st.session_state.partial_edit_query = None
                                
                                st.success("部分編集を適用しました")
                                st.rerun()
                            except Exception as e:
                                st.error(f"生成に失敗: {e}")
                
                if st.button("解釈をやり直す", key="partial_reset_btn"):
                    st.session_state.partial_interpretations = []
                    st.session_state.selected_partial_interpretation_id = None
                    st.rerun()
        else:
            st.info("まずPhase A/Bで画像を用意してください")
    except ImportError:
        st.error("streamlit-drawable-canvas がインストールされていません。")
        st.code("pip install streamlit-drawable-canvas")

    # # --- タブ3: 自由入力（ファジー調整） ---
    # with tab[2]:
    #     fuzzy_text = st.text_input("自由入力（例: もっとサイバーに）", key="fuzzy_text")
    #     default_delta = st.slider("典型的な調整幅（±）", min_value=0.05, max_value=0.6, value=0.25, step=0.05)
    #     if st.button("調整を実行", type="primary"):
    #         try:
    #             system.apply_fuzzy_adjustment(fuzzy_text, default_delta=float(default_delta), auto_generate_image=True)
    #             st.success("調整を適用し、画像を再生成しました")
    #             st.rerun()
    #         except Exception as e:
    #             st.error(f"調整に失敗: {e}")

    # 下部: ノード操作を先に配置（選択されたノードを取得）
    st.markdown("---")
    selected_node_detail = None
    if system.session:
        st.markdown("### ノード操作")
        nodes = system.session.list_nodes()
        if nodes:
            # 現在のノードをデフォルトで選択
            current_idx = 0
            if system.session.current_node_id is not None:
                for i, n in enumerate(nodes):
                    if n['node_id'] == system.session.current_node_id:
                        current_idx = i
                        break
            
            nid_options = [f"#{n['node_id']} (parent {n['parent_id']}, {'CLOSED' if n['is_closed'] else 'open'})" for n in nodes]
            sel = st.selectbox("操作対象ノード", options=nid_options, index=current_idx)
            target_id = int(sel.split(')')[0].split('#')[1].split(' ')[0])
            selected_node_detail = next((n for n in system.session.exploration_nodes if n.node_id == target_id), None)

    # 特徴ベクトルと探索木の表示
    # 選択されたノードの生成画像がある場合は表示
    if selected_node_detail and selected_node_detail.generated_image_path:
        st.subheader(f"ノード #{selected_node_detail.node_id} の生成画像")
        st.image(selected_node_detail.generated_image_path, use_column_width=True)
    
    colv, colt = st.columns([1, 1])
    
    # 選択されたノードの特徴ベクトルを表示
    display_vector = None
    if selected_node_detail:
        display_vector = selected_node_detail.vector
        colv.subheader(f"デザインパラメータ（ノード #{selected_node_detail.node_id}）")
    elif system.session and system.session.current_vector:
        display_vector = system.session.current_vector
        colv.subheader("デザインパラメータ（現在）")
    
    if display_vector:
        df_top = _vector_top_df(display_vector.weights, top_k=10)
        if not df_top.empty:
            chart_df = df_top.set_index("属性")["値"]
            colv.bar_chart(chart_df)
            with colv.expander("詳細（表形式）"):
                colv.dataframe(df_top, use_container_width=True, hide_index=True)

    colt.subheader("ノード履歴（探索木）")
    if gv is not None and system.session and system.session.exploration_nodes:
        dot = _build_tree_dot(system.session)
        colt.graphviz_chart(dot, use_container_width=True)
    else:
        colt.info("グラフ表示にはgraphvizが必要です。")

    # ノード操作ボタン
    if selected_node_detail:
        st.write({
            "node_id": selected_node_detail.node_id,
            "parent_id": selected_node_detail.parent_id,
            "is_closed": selected_node_detail.is_closed,
            "generated_image": selected_node_detail.generated_image_path,
            "note": selected_node_detail.note,
            "timestamp": selected_node_detail.timestamp.isoformat(),
        })
        c1, c2 = st.columns(2)
        if c1.button("このノードに戻る"):
            try:
                system.revert_to_node(selected_node_detail.node_id)
                st.success(f"ノード {selected_node_detail.node_id} に戻りました")
                st.rerun()
            except Exception as e:
                st.error(f"戻りに失敗: {e}")
        if c2.button("このノードをクローズドにする"):
            try:
                system.mark_node_closed(selected_node_detail.node_id)
                st.success(f"ノード {selected_node_detail.node_id} をクローズドにしました")
                st.rerun()
            except Exception as e:
                st.error(f"クローズに失敗: {e}")
    elif system.session and system.session.exploration_nodes:
        st.info("ノードを選択してください。")

    # メインエリア最下部: 属性一覧テーブル
    st.markdown("---")
    st.markdown("### 属性一覧")
    
    # グループごとに属性をテーブル形式で表示
    for gname, items in ATTR_SPACE.groups.items():
        st.subheader(f"🏷️ {gname}")
        # 属性名だけをリスト化
        attr_names = list(items.values())
        # 3カラムに分割してテーブル表示
        cols_data = [[] for _ in range(3)]
        for idx, name in enumerate(attr_names):
            cols_data[idx % 3].append(name)
        
        # 最長カラムの長さに合わせてパディング
        max_len = max(len(col) for col in cols_data)
        for col in cols_data:
            while len(col) < max_len:
                col.append("")
        
        # DataFrameを作成
        table_data = {}
        for i in range(3):
            table_data[f"属性{i+1}"] = cols_data[i]
        df_attrs = pd.DataFrame(table_data)
        st.dataframe(df_attrs, use_container_width=True, hide_index=True)

else:
    st.warning("無効なフェーズです。最初からやり直してください。")

st.markdown("---")
st.caption("© TrueCoding Experimental UI (Streamlit)")
