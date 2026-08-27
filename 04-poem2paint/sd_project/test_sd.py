import torch
from diffusers import StableDiffusionPipeline

# 1. 直接指定你截图中的本地绝对路径，并使用 float16 精度适应你的 V100 显卡
model_path = "/root/autodl-tmp/sd_v15"

pipe = StableDiffusionPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.float16
)

# 2. 将模型加载到 V100 显卡上
pipe = pipe.to("cuda")

# 3. 设定符合你实验背景的古诗水墨画提示词
prompt = "A traditional Chinese ink wash painting, mountains and rivers shrouded in mist, high quality, masterpiece"
print("正在使用本地模型生成图片...")

# 4. 运行生成流水线
image = pipe(prompt).images[0]

# 5. 保存结果
image.save("ink_painting_test.png")
print("图片生成成功！已保存为当前目录下的 ink_painting_test.png")