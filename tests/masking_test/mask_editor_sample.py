# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# from PIL import Image, ImageOps
# import numpy as np
# import io

# def main():
#     st.title("マスク画像作成ツール")
#     st.write("画像をアップロードし、編集したい部分を塗りつぶしてください。")

#     # 1. 画像アップロード
#     uploaded_file = st.file_uploader("背景画像をアップロード", type=["png", "jpg", "jpeg"])

#     if uploaded_file is not None:
#         # 画像を読み込む
#         image = Image.open(uploaded_file).convert("RGB")
        
#         # キャンバスのサイズ設定（画像のサイズに合わせるが、大きすぎる場合は縮小）
#         # 画面からはみ出さないように最大幅を制限
#         max_width = 700
#         canvas_width = min(image.width, max_width)
#         canvas_height = int(image.height * (canvas_width / image.width))
        
#         # 描画設定のカラム分け
#         col1, col2 = st.columns([1, 3])
        
#         with col1:
#             st.subheader("ブラシ設定")
#             # 描画ツールの設定
#             drawing_mode = st.selectbox(
#                 "描画モード",
#                 ("freedraw", "line", "rect", "circle", "transform"),
#                 index=0
#             )
#             stroke_width = st.slider("ブラシサイズ", 1, 100, 20)
            
#             # # マスク作成時は「白」で塗って、背景を「黒」にするのが一般的（またはその逆）
#             # # ここでは視認性を良くするため、半透明の赤などで塗らせて、後で変換する
#             # stroke_color = st.color_picker("ブラシの色", "#FF000088") # 半透明の赤
#             # 【修正1】color_pickerには6桁のHEXコード(#FF0000)を渡す
#             hex_color = st.color_picker("ブラシの色", "#FF0000")
            
#             # 【修正2】キャンバスに渡す際に透明度(88)を付与する (#RRGGBB -> #RRGGBB88)
#             # これにより、ユーザーは不透明色を選んでも、塗るときは半透明になります
#             stroke_color = hex_color + "88"
        
#         with col2:
#             st.subheader("キャンバス")
#             # --- キャンバスの表示 ---
#             canvas_result = st_canvas(
#                 fill_color=stroke_color,  # 塗りつぶし色 (rect/circle用)
#                 stroke_width=stroke_width,
#                 stroke_color=stroke_color,
#                 background_image=image,   # アップロードした画像を背景に設定
#                 update_streamlit=True,    # 描画のたびに更新
#                 height=canvas_height,
#                 width=canvas_width,
#                 drawing_mode=drawing_mode,
#                 key="canvas",
#             )

#         # 2. マスク画像の生成と取得
#         if canvas_result.image_data is not None:
#             st.divider()
#             st.subheader("生成結果")
            
#             # canvas_result.image_data は RGBA の numpy 配列
#             # 描画された部分（透明でない部分）を抽出してマスクを作成する
            
#             # numpy配列からPIL画像に変換
#             mask_data = canvas_result.image_data
            
#             # 描画データのアルファチャンネル(A)を取得 (0=透明, 255=不透明)
#             alpha_channel = mask_data[:, :, 3]
            
#             # アルファチャンネルが0より大きい（描画された）部分を255（白）、それ以外を0（黒）にする
#             # DALL-E 2の仕様では「編集したい部分を透明(または特定の色)にする」必要があるが、
#             # edit_image APIに渡すマスクは「透過部分が編集対象」または「白が編集対象」のバイナリ画像
#             # 一般的に: 白(255) = 編集領域, 黒(0) = 保持領域
            
#             mask_bool = alpha_channel > 0
#             mask_uint8 = mask_bool.astype(np.uint8) * 255
            
#             # PIL Imageとして作成
#             mask_image = Image.fromarray(mask_uint8, mode="L") # L = グレースケール
            
#             # 元の画像サイズにリサイズ（キャンバスで縮小表示していた場合のため）
#             mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)

#             # 結果表示用のカラム
#             res_col1, res_col2 = st.columns(2)
            
#             with res_col1:
#                 st.write("作成されたマスク画像 (白=編集対象)")
#                 st.image(mask_image, use_column_width=True, caption="Mask Image")
                
#                 # ダウンロードボタン
#                 buf = io.BytesIO()
#                 mask_image.save(buf, format="PNG")
#                 byte_im = buf.getvalue()
#                 st.download_button(
#                     label="マスク画像をダウンロード",
#                     data=byte_im,
#                     file_name="mask.png",
#                     mime="image/png"
#                 )

#             with res_col2:
#                 # DALL-E 2 用に「透過マスク画像」も作ってみる（参考）
#                 # マスク部分を透明(0)、それ以外を不透明(255)にした画像
#                 st.write("API送信用プレビュー (透過)")
                
#                 # 元画像をコピー
#                 transparent_image = image.copy()
#                 # アルファチャンネルを追加
#                 transparent_image.putalpha(255)
                
#                 # マスク画像を反転（白黒逆転）させてアルファチャンネルとして適用
#                 # 編集領域(白) -> 反転して黒(0=透明) -> その部分が透明になる
#                 inverted_mask = ImageOps.invert(mask_image)
#                 transparent_image.putalpha(inverted_mask)
                
#                 st.image(transparent_image, use_column_width=True, caption="Inpainting Input")

# if __name__ == "__main__":
#     main()

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import io

def main():
    st.title("マスク画像作成ツール (透過対応版)")
    st.write("画像をアップロードし、**AIに編集させたい部分（消したい部分）**を塗りつぶしてください。")

    # 1. 画像アップロード
    uploaded_file = st.file_uploader("背景画像をアップロード", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # 画像を読み込む
        image = Image.open(uploaded_file).convert("RGB")
        
        # キャンバスのサイズ設定（画像のサイズに合わせるが、大きすぎる場合は縮小）
        max_width = 700
        canvas_width = min(image.width, max_width)
        canvas_height = int(image.height * (canvas_width / image.width))
        
        # 描画設定のカラム分け
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("ブラシ設定")
            drawing_mode = st.selectbox(
                "描画モード",
                ("freedraw", "line", "rect", "circle", "transform"),
                index=0
            )
            stroke_width = st.slider("ブラシサイズ", 1, 100, 20)
            
            # ブラシの色設定（ユーザーには赤で見せるが、内部データとしては不透明度を使う）
            hex_color = st.color_picker("ブラシの色", "#FF0000")
            # キャンバス上の表示用（半透明にして下の絵が見えるようにする）
            stroke_color = hex_color + "88"
        
        with col2:
            st.subheader("キャンバス")
            # --- キャンバスの表示 ---
            canvas_result = st_canvas(
                fill_color=stroke_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                key="canvas",
            )

        # 2. マスク画像の生成と取得
        if canvas_result.image_data is not None:
            st.divider()
            st.subheader("生成結果")
            
            # --- ここからロジック変更 ---
            
            # 1. キャンバスの描画データ（RGBA）を取得
            mask_data = canvas_result.image_data
            
            # 2. アルファチャンネル(A)だけを取り出す
            # 描画された部分には色がつき(A>0)、描画してない部分は透明(A=0)になっている
            alpha_channel = mask_data[:, :, 3]
            
            # 3. ブール配列（True=描画した場所、False=描画してない場所）を作成
            mask_bool = alpha_channel > 0
            
            # 4. 白黒マスク（Lモード）を一時的に作成
            # 描画した場所(True)を255(白)に、それ以外を0(黒)にする
            mask_uint8 = mask_bool.astype(np.uint8) * 255
            temp_mask = Image.fromarray(mask_uint8, mode="L")
            
            # 5. 元の画像サイズにリサイズ（キャンバスの縮小表示対策）
            # ここでリサイズしないと、元の高解像度画像とサイズが合わなくなる
            temp_mask = temp_mask.resize(image.size, Image.Resampling.NEAREST)
            
            # 6. 【重要】API送信用に「透過マスク」を作成する
            # ゴール: 描画した場所 = 透明 (Alpha 0), 背景 = 黒 (Alpha 255)
            
            # まず、全体が「不透明な黒」の画像を作る
            final_mask = Image.new("RGBA", image.size, (0, 0, 0, 255))
            
            # さきほど作った白黒マスクを反転させる (白->黒(0), 黒->白(255))
            # これで「描画した場所が0(透明)」「背景が255(不透明)」のアルファ用データができる
            mask_alpha = ImageOps.invert(temp_mask)
            
            # 作成したアルファ用データを、黒画像に適用する
            final_mask.putalpha(mask_alpha)

            # --- ロジック変更ここまで ---

            # 結果表示用のカラム
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.write("▼ ダウンロード用マスク (市松模様=透明)")
                # 表示用に縮小
                st.image(final_mask, use_column_width=True, caption="Mask for API (Transparent=Edit)")
                
                # ダウンロードボタン
                buf = io.BytesIO()
                final_mask.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="マスク画像をダウンロード (透過PNG)",
                    data=byte_im,
                    file_name="mask_transparent.png",
                    mime="image/png"
                )

            with res_col2:
                st.write("▼ 参考: どの部分が維持されるか")
                # 確認用：黒い部分が維持され、透明部分(背景の白が見える)が編集される
                st.image(final_mask, use_column_width=True, caption="Black = Keep, Transparent = Edit")

if __name__ == "__main__":
    main()