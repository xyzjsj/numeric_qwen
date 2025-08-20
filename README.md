# ��ֵ��ǿQwen2.5-VLģ��

����ԭ��Qwen2.5-VL�ܹ�����ֵ��ǿ��ģ̬��ģ�ͣ�֧�� `<num><value>` ��ʽ����ֵtoken����

## ����

- **ԭ���ܹ�**: ����Qwen2_5_VLForConditionalGeneration�����踴�ӵ����ƴ��
- **��ֵ��ǿ**: ֧��ר�ŵ���ֵtoken `<num><value>` ��ʽ
- **˫�����**: ͬʱ����ı�logits����ֵԤ��
- **�����ʧ**: ��Ͻ�������ʧ�;��������ʧ
- **�˵���ѵ��**: ֧��ֱ�ӵĶ˵���ѵ��������ֽ׶�ѵ��

## �ļ��ṹ

```
/data1/wangzhiye/1a1a11/original/
������ numeric_qwen2_5_vl.py      # ����ģ��ʵ��
������ training_config.py         # ѵ�����ú����ݴ���
������ train.py                   # ѵ���ű�
������ inference.py               # ����ű�
������ prepare_data.py            # ����׼������
������ README.md                  # ˵���ĵ�
������ data/                      # ����Ŀ¼
    ������ numeric_training_data.json
    ������ images/
```

## ���ٿ�ʼ

### 1. ����׼��

```bash
cd /data1/wangzhiye/1a1a11/original
python prepare_data.py
```

�⽫����ʾ��ѵ�����ݺ�ͼ��

### 2. ѵ��ģ��

```bash
python train.py
```

### 3. �������

```bash
python inference.py
```

## ���ݸ�ʽ

ѵ�����ݲ���JSON��ʽ��֧�ֶ��ֶԻ���ͼ��

```json
{
  "id": "sample_001",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n���ͼ���е���ֵ�Ƕ��٣�"
    },
    {
      "from": "gpt", 
      "value": "����ͼ����ʾ����Ҫ��ֵ������\n1. ���۶<num><1234.56>��Ԫ\n2. �����ʣ�<num><15.8>%"
    }
  ],
  "image": "chart_001.jpg"
}
```

## ��ֵToken��ʽ

ʹ�� `<num><value>` ��ʽ��ʾ��ֵ��

- `<num><3.14159>` - ��ʾ�еĽ���ֵ
- `<num><-273.15>` - ��ʾ�������
- `<num><1.618>` - ��ʾ�ƽ����

## ģ�ͼܹ�

### �������

1. **NumericQwen2_5_VLConfig**: ��չ������
   - �����ֵ������ز���
   - ������ԭ��Qwen2.5-VL�ļ�����

2. **NumericQwen2_5_VLProcessor**: ��ֵ��ǿ������
   - �Զ���ȡ�ı��е���ֵtoken
   - �ṩ��ֵλ����Ϣ

3. **NumericQwen2_5_VLForConditionalGeneration**: ����ģ����
   - ��ֵǶ�����磺1 �� 512 �� hidden_size
   - �ع�ͷ��hidden_size �� 1
   - �����ʧ����

### ��ʧ����

```python
total_loss = token_loss + �� * numeric_loss
```

- `token_loss`: ��������ʧ���ı����ɣ�
- `numeric_loss`: ���������ʧ����ֵԤ�⣩
- `��`: ��ֵ��ʧȨ�أ�Ĭ��1.0��

## ���ò���

### ѵ������

```python
training_args = NumericTrainingArguments(
    output_dir="./output",
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    
    # ѧϰ��
    learning_rate=1e-5,
    vision_lr=2e-6,       # �Ӿ�������ѧϰ��
    numeric_lr=1e-4,      # ��ֵ��ѧϰ��
    
    # ��ֵ�ض�����
    numeric_loss_weight=1.0,
    
    # ѵ������
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
    gradient_checkpointing=True
)
```

### ģ�Ͳ���

```python
numeric_config = {
    'numeric_embedding_dim': 512,    # ��ֵǶ��ά��
    'numeric_token': '<num>',        # ��ֵtoken
    'numeric_loss_weight': 1.0       # ��ʧȨ��
}
```

## ʹ��ʾ��

### ѵ���Զ�������

```python
from training_config import create_model_and_processor, NumericDataset

# ����ģ��
model, processor = create_model_and_processor(
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    numeric_config={'numeric_loss_weight': 1.5}
)

# �������ݼ�
dataset = NumericDataset(
    data_path="your_data.json",
    processor=processor,
    image_folder="your_images/"
)
```

### ����ʹ��

```python
from inference import NumericQwen2_5_VLInference

# ������������
inference = NumericQwen2_5_VLInference("./output")

# �ı�����
result = inference.generate_response("�е�ֵ�Ƕ��٣�")
print(result['text'])  # "�еĽ���ֵ��<num><3.14159>"
print(result['numeric_predictions'])  # [{"value": 3.14159, ...}]

# ͼ������
result = inference.generate_response(
    "�������ͼ���е�����", 
    image="chart.jpg"
)
```

## �����Ż�

### DeepSpeed����

�Զ�����DeepSpeed ZeRO-2���ã�

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

### �ڴ��Ż�

- ʹ��`bf16`����ѵ��
- ����`gradient_checkpointing`
- ֧��ZeRO�Ż���״̬��Ƭ

## ��ԭʼʵ�ֵĶԱ�

| ���� | ԭʼcustom_qwen.py | ��ʵ�� |
|------|------------------|--------|
| �����ܹ� | ��ͷʵ��Qwen2.5-VL | �̳�ԭ��Qwen2_5_VLForConditionalGeneration |
| ������ | ��Ҫ�ֶ����� | ��ȫ����Transformers |
| ѵ�����Ӷ� | ��Ҫ��ȶ��� | ��׼ѵ������ |
| ά���ɱ� | �� | �� |
| ��չ�� | ���� | ���� |

## ����Ҫ��

```bash
# ����conda����
conda activate llava

# ��װ����
pip install torch transformers accelerate deepspeed wandb
pip install pillow datasets
```

## �����ų�

### ��������

1. **CUDA�ڴ治��**
   - ����`per_device_train_batch_size`
   - ����`gradient_accumulation_steps`
   - ����DeepSpeed ZeRO-3

2. **��ֵtokenδ��ʶ��**
   - ���tokenizer�Ƿ���ȷ�����`<num>`token
   - ��֤`num_token_id`����

3. **��ʧ������**
   - ����`numeric_loss_weight`
   - ���ѧϰ������
   - ��֤���ݸ�ʽ

### ����ģʽ

��ѵ���ű������õ�����Ϣ��

```python
# ��ѵ�������л��ӡ��ʧ����
# Token Loss: 2.3456, Float Loss: 0.1234
```

## ����ָ��

1. Fork��Ŀ
2. �������Է�֧
3. �ύ����
4. ����Pull Request

## ���֤

Apache 2.0 License

## ��л

- ����Qwen2.5-VLԭ���ܹ�
- �ο�LLaVA-NeXTѵ�����
- ��лTransformers���ǿ��֧��
