import os
from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor
from PIL import Image, ImageDraw
import torch

"""最简图像问答测试：使用单个 <|image_pad|> 占位符，避免手动重复导致索引错误"""

model_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
img_path = "test_image.png"
question = "描述一下这张图片的内容。"

if not os.path.exists(img_path):
    img = Image.new('RGB', (224, 224), 'white')
    d = ImageDraw.Draw(img)
    d.rectangle([50, 50, 150, 100], fill='blue')
    d.ellipse([100, 120, 180, 200], fill='red')
    img.save(img_path)

image = Image.open(img_path)

model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
processor = NumericQwen2_5_VLProcessor.from_pretrained(model_path)

# 单图像占位符（让处理器自己注入正确数量的视觉特征与 grid 信息）
text_prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

print("[DEBUG] input_ids.shape:", inputs.input_ids.shape)
if 'image_grid_thw' in inputs:
    print("[DEBUG] image_grid_thw:", inputs.image_grid_thw)
if 'pixel_values' in inputs:
    print("[DEBUG] pixel_values.shape:", inputs.pixel_values.shape)
image_token_id = getattr(model.config, 'image_token_id', None)
if image_token_id is not None:
    count_image_tokens = (inputs.input_ids == image_token_id).sum().item()
    print(f"[DEBUG] counted image tokens before generation: {count_image_tokens}")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=processor.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
    )

generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("回答:", answer[0])
