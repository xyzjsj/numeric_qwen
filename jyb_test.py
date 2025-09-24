from jyb_numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_PATH = "/data1/wangzhiye/qwen253B"
model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    # "/data1/wangzhiye/qwen253B",
    MODEL_PATH,
    # "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250",
    device_map="auto")
processor = NumericQwen2_5_VLProcessor.from_pretrained(MODEL_PATH,local_files_only=True)

# 帮我生成一个使用模型generate的案例
conversation = [
    {"role": "user", "content": [
        {"type": "text", "text": "这个产品评分为<num><8.5>分，价格是<num><299.99>元。"}
    ]},
]
text = processor.apply_chat_template(conversation,
                                    # return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    add_generation_prompt=True)
inputs = processor(text=text, images=None, return_tensors="pt", padding=True)
print("inputs:", inputs)
inputs = inputs.to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

