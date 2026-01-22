# 新規部分編集セクションの更新版コード（app.pyの該当部分を置き換える）

# セッション状態初期化に追加
if "partial_interpretations" not in st.session_state:
    st.session_state.partial_interpretations = []
    st.session_state.selected_partial_interpretation_id = None
    st.session_state.partial_edit_mask_path = None
    st.session_state.partial_edit_target_name = None
    st.session_state.partial_edit_base_path = None
    st.session_state.partial_edit_query = None

# メインエリア: 新規部分編集セクション（2段階フロー）
st.markdown("---")
st.subheader("✂️ 新規部分編集")
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
                            part_text = f"{st.session_state.partial_edit_target_name}: {st.session_state.partial_edit_query}"
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
