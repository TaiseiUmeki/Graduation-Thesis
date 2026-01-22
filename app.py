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
    images_dir = Path("data/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix.lower() or ".jpg"
    fname = f"{prefix}_{uuid.uuid4().hex}{ext}"
    fpath = images_dir / fname
    with open(fpath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(fpath)

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
                base_img = Image.open(base_path)
                canvas_width = min(base_img.width, 800)
                canvas_height = int(base_img.height * (canvas_width / base_img.width))
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=20,
                    stroke_color="#FF0000",
                    background_image=Image.open(base_path),
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
                            mask_path = Path("data/images") / f"mask_{uuid.uuid4().hex}.png"
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
                                            constraints=[],
                                            max_attrs=8
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
                                        edit_intent=st.session_state.phase_a_partial_query
                                    )
                                    
                                    # 子ノード作成
                                    from models.session import GeneratedImage
                                    system.session.add_child_node(
                                        vector=global_vec,
                                        constraints=system.session.constraints,
                                        note=f"partial_edit:{st.session_state.phase_a_partial_target_name}",
                                        partial_vector=partial_vec,
                                        mask_image_path=st.session_state.phase_a_partial_mask_path,
                                        target_part_name=st.session_state.phase_a_partial_target_name
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
        
        try:
            from streamlit_drawable_canvas import st_canvas
            
            synthesis_up = st.file_uploader(
                "合成元画像をアップロード",
                type=["png", "jpg", "jpeg"],
                key="synthesis_uploader"
            )
            
            if synthesis_up:
                synthesis_base_path = _save_uploaded_image(synthesis_up, prefix="synthesis_base")
                st.image(synthesis_base_path, caption="合成元画像", use_column_width=True)
                
                st.markdown("#### 結合位置をマスクで指定")
                st.caption("赤色で結合したい領域を塗りつぶしてください")
                
                # Canvas for drawing mask
                from PIL import Image as PILImage
                base_img = PILImage.open(synthesis_base_path)
                canvas_width = min(base_img.width, 800)
                canvas_height = int(base_img.height * (canvas_width / base_img.width))
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=20,
                    stroke_color="#FF0000",
                    background_image=PILImage.open(synthesis_base_path),
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="freedraw",
                    key="canvas",
                )
                
                synthesis_instruction = st.text_input(
                    "結合指示",
                    value="右のパーツを砲台として結合して",
                    key="synthesis_instruction"
                )
                
                if st.button("合成実行", type="primary", key="synthesis_btn"):
                    if canvas_result.image_data is not None:
                        with st.spinner("粘土パーツを合成中..."):
                            # マスク画像を透過形式で作成（mask_editor_sample.py の処理を参照）
                            # 1. キャンバスの描画データ（RGBA）を取得
                            mask_data = canvas_result.image_data
                            
                            # 2. アルファチャンネル(A)だけを取り出す
                            alpha_channel = mask_data[:, :, 3]
                            
                            # 3. ブール配列（True=描画した場所、False=描画してない場所）を作成
                            mask_bool = alpha_channel > 0
                            
                            # 4. 白黒マスク（Lモード）を一時的に作成
                            mask_uint8 = mask_bool.astype(np.uint8) * 255
                            temp_mask = Image.fromarray(mask_uint8, mode="L")
                            
                            # 5. 元の画像サイズにリサイズ（キャンバスの縮小表示対策）
                            temp_mask = temp_mask.resize(base_img.size, Image.Resampling.NEAREST)
                            
                            # 6. 【重要】API送信用に「透過マスク」を作成する
                            # ゴール: 描画した場所 = 透明 (Alpha 0), 背景 = 黒 (Alpha 255)
                            
                            # まず、全体が「不透明な黒」の画像を作る
                            final_mask = Image.new("RGBA", base_img.size, (0, 0, 0, 255))
                            
                            # さきほど作った白黒マスクを反転させる (白->黒(0), 黒->白(255))
                            # これで「描画した場所が0(透明)」「背景が255(不透明)」のアルファ用データができる
                            mask_alpha = ImageOps.invert(temp_mask)
                            
                            # 作成したアルファ用データを、黒画像に適用する
                            final_mask.putalpha(mask_alpha)
                            
                            # 透過マスクを保存
                            mask_path = Path("data/images") / f"mask_{uuid.uuid4().hex}.png"
                            final_mask.save(mask_path)
                            
                            try:
                                # 合成実行
                                result_path = system.image_generator.generate_synthesis(
                                    synthesis_base_path,
                                    str(mask_path),
                                    synthesis_instruction
                                )
                                st.session_state.synthesis_result_path = result_path
                                st.success("合成が完了しました！")
                                st.image(result_path, caption="合成結果", use_column_width=True)
                                
                                if st.button("この画像を使用", key="use_synthesis"):
                                    st.session_state.uploaded_image_path = result_path
                                    st.success("合成画像を入力画像として設定しました")
                            except Exception as e:
                                st.error(f"合成に失敗: {e}")
                    else:
                        st.warning("マスクを描画してください")
        
        except ImportError:
            st.error("streamlit-drawable-canvas がインストールされていません。")
            st.code("pip install streamlit-drawable-canvas")
    
    # --- モチーフ（Concept）とクエリ入力 ---
    st.markdown("---")
    st.subheader("モチーフと意図を入力")
    
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
        min_val, max_val = st.sidebar.slider("値の範囲", min_value=-1.0, max_value=1.0, value=(-0.1, 0.1), step=0.05)
    else:
        val = st.sidebar.slider("値", min_value=-1.0, max_value=1.0, value=0.3, step=0.05)

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
            # 再生成（グローバル変換）
            system.generate_image()
            st.success("画像を再生成しました")
            st.rerun()
        except Exception as e:
            st.error(f"制約の適用に失敗: {e}")

    # サイドバー: 微調整セクション
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 微調整（現在選択ノードが部分編集の場合）")
    try:
        # 現在ノードを取得
        selected_node_detail_for_tuning = None
        if system.session and system.session.current_node_id is not None:
            selected_node_detail_for_tuning = next((n for n in system.session.exploration_nodes if n.node_id == system.session.current_node_id), None)
        if selected_node_detail_for_tuning and getattr(selected_node_detail_for_tuning, "partial_vector", None):
            pv = selected_node_detail_for_tuning.partial_vector
            # 上位属性をスライダーで調整
            from models.attribute_space import AttributeVector
            top_items = sorted(pv.weights.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            new_weights = dict(pv.weights)
            for attr_key, weight in top_items:
                name = ATTR_SPACE.get_attribute_name(attr_key) or attr_key
                new_weights[attr_key] = st.sidebar.slider(f"{name} ({attr_key})", min_value=-1.0, max_value=1.0, value=float(weight), step=0.05, key=f"pv_{selected_node_detail_for_tuning.node_id}_{attr_key}")

            if st.sidebar.button("この部分ベクトルで再編集", key=f"pv_apply_{selected_node_detail_for_tuning.node_id}"):
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
                        new_pv = AttributeVector(weights=new_weights)
                        result_path = system.image_generator.generate_part_from_vector(
                            base_image_path=base_path,
                            mask_path=str(selected_node_detail_for_tuning.mask_image_path),
                            partial_vector=new_pv,
                            concept=system.session.concept,
                            target_part_name=selected_node_detail_for_tuning.target_part_name
                        )
                        from models.session import GeneratedImage
                        system.session.add_child_node(
                            vector=selected_node_detail_for_tuning.vector,
                            constraints=system.session.constraints,
                            note=f"partial_tune:{selected_node_detail_for_tuning.target_part_name}",
                            partial_vector=new_pv,
                            mask_image_path=selected_node_detail_for_tuning.mask_image_path,
                            target_part_name=selected_node_detail_for_tuning.target_part_name
                        )
                        system.session.update_current_node_image(result_path)
                        system.session.add_generated_image(GeneratedImage(image_path=result_path, prompt="partial_tune", vector=new_pv, constraints=system.session.constraints))
                        st.success("部分ベクトルを反映して再編集しました")
                        st.rerun()
        else:
            st.sidebar.info("部分編集ノードを選択すると微調整UIが表示されます")
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
                base_img = Image.open(base_path)
                canvas_width = 300
                canvas_height = int(base_img.height * (canvas_width / base_img.width))
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=15,
                    stroke_color="#FF0000",
                    background_image=Image.open(base_path),
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
                        mask_path = Path("data/images") / f"mask_{uuid.uuid4().hex}.png"
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
                                
                                # 部分ベクトル生成（選択された解釈から）
                                if system.session.selected_interpretation:
                                    partial_vec = system.vector_generator.generate_vector_from_interpretation(
                                        system.session.selected_interpretation,
                                        constraints=[],
                                        max_attrs=8
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
                                    edit_intent=st.session_state.partial_edit_query
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
                                    target_part_name=st.session_state.partial_edit_target_name
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
