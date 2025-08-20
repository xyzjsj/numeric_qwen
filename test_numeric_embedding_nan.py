import torch
from numeric_qwen2_5_vl import NumericQwen2_5_VLConfig, NumericQwen2_5_VLForConditionalGeneration

# 构造一个简单的 config
config = NumericQwen2_5_VLConfig(
    numeric_embedding_dim=16,
    hidden_size=32,
    num_attention_heads=4,  # 必须指定，否则 head_dim=0
    num_hidden_layers=2,    # 层数也补上
    vocab_size=1000         # 随便给个小词表，避免初始化报错
)

# 实例化模型
model = NumericQwen2_5_VLForConditionalGeneration(config)

# 构造一组测试数值（包含正常值、极端值、nan、inf）
test_values = torch.tensor([[489.0], [0.0], [-23421.0], [1e10], [-1e10]])

# 测试 embedding 层输出
with torch.no_grad():
    emb = model.numeric_embedding(test_values)
    print("输入:", test_values.squeeze().tolist())
    print("embedding 输出:")
    print(emb)
    print("embedding 是否包含 nan:", torch.isnan(emb).any().item())
    print("embedding 是否包含 inf:", torch.isinf(emb).any().item())
    print("embedding 最小值:", emb.min().item())
    print("embedding 最大值:", emb.max().item())
    print("embedding 均值:", emb.mean().item())
    print("embedding 标准差:", emb.std().item())
