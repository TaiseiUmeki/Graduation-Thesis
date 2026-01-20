import os
import uuid
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd

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

# ページ設定
st.set_page_config(page_title="梅木卒論 GUI", layout="wide")

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

system: TrueCodingSystem = st.session_state.system
phase: str = st.session_state.phase

# サイドバー: 操作パネル
st.sidebar.title("操作パネル")
st.title(" 梅木卒論 実験用GUI")

# ========== Phase A: 初期入力 ==========
if phase == "A":
    st.sidebar.subheader("Phase A: 初期入力")
    up = st.sidebar.file_uploader("粘土画像をアップロード", type=["png", "jpg", "jpeg"]) 
    query = st.sidebar.text_input("初期の意図（クエリ）", value="")
    start_clicked = st.sidebar.button("セッション開始", type="primary")

    if start_clicked:
        if not up or not query.strip():
            st.sidebar.error("画像とクエリを入力してください")
        else:
            img_path = _save_uploaded_image(up, prefix="initial")
            try:
                system.start_session(image_path=img_path, query=query.strip())
                st.session_state.phase = "B"
                st.session_state.interpretations = []
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"開始に失敗: {e}")

    st.info("サイドバーから画像とクエリを入力し、セッションを開始してください。")

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
    col1, col2 = st.columns(2)
    if system.session and system.session.initial_image_path:
        col1.subheader("元画像")
        col1.image(system.session.initial_image_path, use_column_width=True)
    latest = system.session.get_latest_image() if system.session else None
    if latest:
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

    # サイドバー: タブ構成
    st.sidebar.subheader("調整タブ")
    tab = st.sidebar.tabs(["属性検索", "こだわり条件（制約）"]) 

    # --- タブ1: 属性検索 ---
    with tab[0]:
        q = st.text_input("属性検索キーワード", key="attr_search_q")
        max_k = st.number_input("最大件数", min_value=1, max_value=20, value=5)
        if st.button("検索", key="attr_search_btn"):
            try:
                res = system.search_attributes(q, max_results=int(max_k), return_expanded=True)
                st.session_state.search_results = res or []
            except Exception as e:
                st.error(f"検索に失敗: {e}")
        if st.session_state.search_results:
            st.markdown("#### 検索結果")
            df = pd.DataFrame(st.session_state.search_results)
            if not df.empty:
                st.dataframe(df, use_container_width=True)

    # --- タブ2: 制約で調整 ---
    with tab[1]:
        st.markdown("#### 制約を追加")
        # セレクトボックス用オプション（グループ順を保持）
        all_attrs: List[Tuple[str, str]] = []  # (display, key)
        for gname, items in ATTR_SPACE.groups.items():
            for k, v in items.items():
                key = f"{gname}:{k}"
                display = f"{v} ({key})"
                all_attrs.append((display, key))
        # ソートを削除（グループ順を保持）
        display_labels = [d for d, k in all_attrs]
        selected_display = st.selectbox("属性キー", options=display_labels)
        selected_key = dict(all_attrs)[selected_display]

        ctype_map = {
            "≥ 以上": "greater_than",
            "≤ 以下": "less_than",
            "= 等しい": "equal",
            "範囲指定": "range",
        }
        ctype_label = st.radio("制約タイプ", options=list(ctype_map.keys()), horizontal=True)
        ctype = ctype_map[ctype_label]

        val = None
        min_val = None
        max_val = None
        if ctype == "range":
            min_val, max_val = st.slider("値の範囲", min_value=-1.0, max_value=1.0, value=(-0.1, 0.1), step=0.05)
        else:
            val = st.slider("値", min_value=-1.0, max_value=1.0, value=0.3, step=0.05)

        desc = st.text_input("説明（任意）", value="")
        if st.button("制約を追加して再生成", type="primary"):
            try:
                system.add_constraint(
                    attribute_key=selected_key,
                    constraint_type=ctype,
                    value=float(val) if val is not None else None,
                    min_value=float(min_val) if min_val is not None else None,
                    max_value=float(max_val) if max_val is not None else None,
                    description=desc,
                )
                # 再生成
                system.generate_image()
                st.success("画像を再生成しました")
                st.rerun()
            except Exception as e:
                st.error(f"制約の適用に失敗: {e}")

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
