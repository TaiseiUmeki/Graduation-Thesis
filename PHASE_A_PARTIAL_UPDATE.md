# Phase A 部分編集モードの更新版コード

## セッション状態初期化に追加（95-105行目の後）

```python
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
    # Phase A 部分編集用の状態
    st.session_state.phaseA_partial_interpretations = []
    st.session_state.phaseA_partial_mask_path = None
    st.session_state.phaseA_partial_target_name = None
    st.session_state.phaseA_partial_base_path = None
    st.session_state.phaseA_partial_query = None
    st.session_state.phaseA_partial_concept = None
```

## Phase A タブ2の置き換え（131-244行目）

```python
    # --- タブ2: 部分編集モード ---
    with input_tab2:
        st.subheader("部分編集（インペインティング）")

        # Phase A 部分編集用の状態初期化
        if "phaseA_partial_interpretations" not in st.session_state:
            st.session_state.phaseA_partial_interpretations = []
            st.session_state.phaseA_partial_mask_path = None
            st.session_state.phaseA_partial_target_name = None
            st.session_state.phaseA_partial_base_path = None
            st.session_state.phaseA_partial_query = None
            st.session_state.phaseA_partial_concept = None

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
                concept_partial = colp1.text_input("モチーフ (例: tank)", value="", key="phaseA_partial_concept_input")
                target_name = colp2.text_input("部位名 (例: turret)", value="", key="phaseA_partial_target_input")
                partial_query = colp3.text_input("編集意図（クエリ）", value="", key="phaseA_partial_query_input")

                # ステップ1: 意図を解釈
                if st.button("意図を解釈（Interpret）", type="primary", key="phaseA_partial_interpret"):
                    if not (concept_partial.strip() and target_name.strip() and partial_query.strip()):
                        st.warning("モチーフ・部位名・編集意図を入力してください")
                    elif canvas_result.image_data is None:
                        st.warning("マスクを描画してください")
                    else:
                        with st.spinner("編集意図を解釈中..."):
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

                            # セッション状態に保存
                            st.session_state.phaseA_partial_mask_path = str(mask_path)
                            st.session_state.phaseA_partial_target_name = target_name.strip()
                            st.session_state.phaseA_partial_base_path = base_path
                            st.session_state.phaseA_partial_query = partial_query.strip()
                            st.session_state.phaseA_partial_concept = concept_partial.strip()

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
                                st.session_state.phaseA_partial_interpretations = [(i.id, i.text, i.reasoning) for i in interps]
                                st.success("解釈を生成しました")
                                st.rerun()
                            except Exception as e:
                                st.error(f"解釈生成に失敗: {e}")

                # ステップ2: 解釈を選択して生成実行
                if st.session_state.phaseA_partial_interpretations:
                    st.markdown("---")
                    st.subheader("📋 解釈候補を選択")
                    options = [f"[{i}] {t}" for (i, t, r) in st.session_state.phaseA_partial_interpretations]
                    choice = st.radio("解釈を選んでください", options=options, index=0, key="phaseA_partial_choice")
                    chosen_id = int(choice.split(']')[0][1:]) if choice else None

                    with st.expander("根拠（Reasoning）"):
                        for (i, t, r) in st.session_state.phaseA_partial_interpretations:
                            st.markdown(f"- [{i}] {r}")

                    col_gen, col_reset = st.columns(2)
                    if col_gen.button("生成実行", type="primary", key="phaseA_partial_generate"):
                        if chosen_id is None:
                            st.warning("解釈を選択してください")
                        else:
                            with st.spinner("Root作成と部分編集を実行中..."):
                                try:
                                    # 選択された解釈を適用
                                    system.select_interpretation(chosen_id)

                                    # Rootノード（グローバルベクトル）を画像分析から生成
                                    global_vec = system.vector_generator.generate_global_from_image(
                                        st.session_state.phaseA_partial_base_path,
                                        concept=st.session_state.phaseA_partial_concept
                                    )
                                    system.session.set_vector(global_vec)
                                    system.session.add_root_node(global_vec, system.session.constraints, note="root")

                                    # 部分ベクトル生成（選択された解釈から）
                                    part_text = f"{st.session_state.phaseA_partial_target_name}: {st.session_state.phaseA_partial_query}"
                                    partial_vec = system.vector_generator.generate_from_text(
                                        part_text,
                                        concept=st.session_state.phaseA_partial_concept,
                                        max_attrs=8
                                    )

                                    # インペインティングで部分編集
                                    result_path = system.image_generator.generate_part_from_vector(
                                        base_image_path=st.session_state.phaseA_partial_base_path,
                                        mask_path=st.session_state.phaseA_partial_mask_path,
                                        partial_vector=partial_vec,
                                        concept=st.session_state.phaseA_partial_concept,
                                        target_part_name=st.session_state.phaseA_partial_target_name,
                                        edit_intent=st.session_state.phaseA_partial_query
                                    )

                                    # 子ノード作成
                                    from models.session import GeneratedImage
                                    system.session.add_child_node(
                                        vector=global_vec,
                                        constraints=system.session.constraints,
                                        note=f"partial_edit:{st.session_state.phaseA_partial_target_name}",
                                        partial_vector=partial_vec,
                                        mask_image_path=st.session_state.phaseA_partial_mask_path,
                                        target_part_name=st.session_state.phaseA_partial_target_name
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
                                    st.session_state.phaseA_partial_interpretations = []
                                    st.session_state.phaseA_partial_mask_path = None
                                    st.session_state.phaseA_partial_target_name = None
                                    st.session_state.phaseA_partial_base_path = None
                                    st.session_state.phaseA_partial_query = None
                                    st.session_state.phaseA_partial_concept = None

                                    st.success("部分編集を完了しました！")
                                    st.image(result_path, caption="部分編集結果", use_column_width=True)
                                    st.session_state.phase = "D"
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"生成に失敗: {e}")

                    if col_reset.button("解釈をやり直す", key="phaseA_partial_reset"):
                        st.session_state.phaseA_partial_interpretations = []
                        st.rerun()

        except ImportError:
            st.error("streamlit-drawable-canvas がインストールされていません。")
            st.code("pip install streamlit-drawable-canvas")
```
