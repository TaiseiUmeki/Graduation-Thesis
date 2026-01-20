import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Stable Diffusion の読み込み
model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
# ここが変更点: Img2Img用のパイプラインを使う
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# 2. 粘土解析用のモデルも「全く同じもの」を読み込む (ここ重要)
# SD v1.5 は CLIP ViT-L/14 を使っています
clip_model_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

# ---------------------------------------------------------
# A. 粘土から差分ベクトルを抽出 (前の実験と同じロジック)
# ---------------------------------------------------------
# image_before, image_after は PIL Image とします
# ※ SD v1.5 のテキスト次元数は 768 です
with torch.no_grad():
    inputs = clip_processor(images=[image_before, image_after], return_tensors="pt").to(device)
    img_features = clip_model.get_image_features(**inputs)
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    
    # 差分ベクトル (言葉にできないこだわり)
    clay_diff_vector = img_features[1] - img_features[0] 
    
    # 正規化
    clay_diff_vector = clay_diff_vector / clay_diff_vector.norm(dim=-1, keepdim=True)

# ---------------------------------------------------------
# B. ベースとなるプロンプトをベクトル化
# ---------------------------------------------------------
# 本番は特徴ベクトル
prompt = "a wooden desk, photorealistic, 4k"
# パイプラインの関数を使って、テキストを埋め込みベクトル(prompt_embeds)に変換
# これにより (Batch_Size, 77, 768) の形状のテンソルが得られます
base_embeds, _ = pipe.encode_prompt(
    prompt=prompt, 
    device=device, 
    num_images_per_prompt=1, 
    do_classifier_free_guidance=True,
    negative_prompt=""
)

# ---------------------------------------------------------
# C. ベクトルの注入 (Injection)
# ---------------------------------------------------------
# 差分ベクトルをプロンプト埋め込みに足し合わせる
# clay_diff_vector は (1, 768) なので、base_embeds の全トークン、あるいは特定のトークンに加算します
# ここではシンプルに全体に影響を与えるように足します
alpha = 0.8  # こだわりの強さ

# 次元を合わせるためのunsqueezeなどは状況に合わせて調整
# base_embeds shape: [1, 77, 768]
# diff shape needs to broadcast
injection_vector = clay_diff_vector.view(1, 1, -1) 

# ★ 核心部分: 言葉のベクトルに、粘土のベクトルを直接足す
modified_embeds = base_embeds + (alpha * injection_vector)

# ---------------------------------------------------------
# D. 画像生成
# ---------------------------------------------------------
# ユーザーが編集したい「元画像」を読み込みます
from PIL import Image
init_image = Image.open("path/to/user_original_image.jpg").convert("RGB")
init_image = init_image.resize((512, 512)) # SDは512x512が基本

# ここが重要: strength (強さ) パラメータ
# 0.0 に近いほど元画像そのまま、1.0 に近いほどプロンプト（粘土）の影響が強くなります。
# 0.6 〜 0.8 くらいが「元の形を留めつつ編集する」良い塩梅です。
strength = 0.75

# prompt引数の代わりに prompt_embeds を使い、さらに image を渡します
image = pipe(
    prompt_embeds=modified_embeds,  # 粘土のニュアンス入りベクトル
    image=init_image,               # 元画像
    strength=strength,              # 変化の度合い
    guidance_scale=7.5,             # プロンプトへの従いやすさ
    num_inference_steps=50          # ステップ数は少し多めが良い
).images[0]

image.save("result_img2img_with_clay.png")

# # prompt引数ではなく、prompt_embeds引数を使って生成できるのが SD の強みです
# image = pipe(
#     prompt_embeds=modified_embeds, 
#     # negative_prompt_embeds=... (必要ならこっちも計算)
#     guidance_scale=7.5,
#     num_inference_steps=30
# ).images[0]

# image.save("result_with_clay_nuance.png")
