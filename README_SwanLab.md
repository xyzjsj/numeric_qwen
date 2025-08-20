# ��ֵ��ǿQwen2.5-VLģ�� - SwanLab���ӻ�ѵ��

����Ŀ������SwanLab���ӻ����ߣ�����ʵʱ��غͷ�����ֵ��ǿQwen2.5-VLģ�͵�ѵ�����̡�

## ? SwanLab��Ŀ��Ϣ

- **��Ŀ����**: `qsinghua`
- **�鿴��ַ**: https://swanlab.cn/qsinghua
- **�����ص�**: ѵ����ʧ���ӻ���ģ�Ͳ�����ء���������չʾ��ʵ��Ա�

## ? ���ٿ�ʼ

### 1. ����׼��

ȷ���Ѱ�װSwanLab��
```bash
pip install swanlab
```

��¼SwanLab�˺ţ�
```bash
swanlab login
# ����Python�е�¼
# import swanlab
# swanlab.login(api_key="your_api_key")
```

### 2. ����SwanLab����

�ڿ�ʼ��ʽѵ��ǰ�������Ȳ���SwanLab�����Ƿ�������

```bash
python test_swanlab.py
```

��ʹ�������ű��Ĳ���ģʽ��
```bash
python start_swanlab_training.py --test_only
```

### 3. �������ӻ�ѵ��

#### ��ʽһ��ʹ�������ű����Ƽ���

```bash
# ʹ��Ĭ����������ѵ��
python start_swanlab_training.py

# �Զ�������
python start_swanlab_training.py \
    --data_path /path/to/your/data.json \
    --image_folder /path/to/images \
    --output_dir /path/to/output \
    --swanlab_project qsinghua \
    --swanlab_experiment my_experiment_v1 \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 2e-5
```

#### ��ʽ����ֱ������ѵ���ű�

```bash
python train.py
```

## ? SwanLab���ӻ�����

### ѵ�����ָ��

ѵ�������л��Զ���¼����ָ�굽SwanLab��

1. **��ʧָ��**
   - `train/loss` - ����ѵ����ʧ
   - `loss/total_loss` - ��ϸ����ʧ
   - `loss/numeric_loss` - ��ֵ��ʧ���
   - `loss/language_loss` - ����ģ����ʧ���

2. **ѵ������**
   - `train/epoch` - ��ǰѵ������
   - `train/learning_rate` - ��ǰѧϰ��
   - `time/elapsed_time` - ��ѵ��ʱ��
   - `time/step_time` - ����ѵ��ʱ��

3. **ģ����Ϣ**
   - `model/total_parameters` - ģ���ܲ�����
   - `model/trainable_parameters` - ��ѵ��������
   - `model/vocab_size` - �ʻ���С
   - `model/num_token_id` - ��ֵtoken ID

4. **ѵ������**
   - ����ѵ��������
   - ���ݼ���Ϣ
   - ģ�ͼܹ�����

5. **��������չʾ**
   - ѵ����������Ԥ��
   - ��ֵ��ע��Ϣ
   - ͼ������Ϣ

### ʵ�������

- **ʵ��Ա�**: ��SwanLab�����жԱȲ�ͬʵ�������
- **ʵʱ���**: Զ�̲鿴ѵ�����ȣ�֧���ֻ��鿴
- **Ӳ�����**: �Զ���¼GPU��CPU���ڴ�ʹ�����
- **��־��¼**: ������ѵ����־��¼
- **ģ�Ͱ汾����**: �Զ���¼������Ϣ

## ?? ����ѡ��

### ѵ������

�� `training_config.py` �п�������SwanLab��ز�����

```python
@dataclass
class NumericTrainingArguments(TrainingArguments):
    # SwanLab���ӻ�����
    swanlab_project: str = "qsinghua"  # SwanLab��Ŀ����
    swanlab_experiment: str = None     # ʵ�����ƣ��Զ����ɣ�
    enable_swanlab: bool = True        # �Ƿ�����SwanLab
```

### �����ű�ѡ��

```bash
python start_swanlab_training.py --help
```

���ò�����
- `--swanlab_project`: SwanLab��Ŀ���ƣ�Ĭ��: qsinghua��
- `--swanlab_experiment`: ʵ�����ƣ�Ĭ��: �Զ�����ʱ�����
- `--disable_swanlab`: ����SwanLab���ӻ�
- `--test_only`: ������SwanLab����
- `--data_path`: ѵ������·��
- `--epochs`: ѵ������
- `--batch_size`: ���δ�С
- `--learning_rate`: ѧϰ��

## ? �鿴ѵ�����

### ���߲鿴

���� https://swanlab.cn/qsinghua �鿴����ʵ������

### ��Ҫ�������

1. **ѵ������**
   - ��ʧ�����仯����
   - ѧϰ�ʵ�������
   - ѵ��ʱ�����

2. **ģ�ͷ���**
   - ������ͳ��
   - ��ֵtoken����Ч��
   - �ʻ����չ���

3. **���ݷ���**
   - ���ݼ������ֲ�
   - ��ֵ��עͳ��
   - ͼ�������

4. **Ӳ������**
   - GPU������
   - �ڴ�ʹ�����
   - ѵ��Ч�ʷ���

## ? �����ų�

### ��������

1. **SwanLab����ʧ��**
   ```bash
   # �����������
   ping swanlab.cn
   
   # ���µ�¼
   swanlab login
   ```

2. **ָ���¼ʧ��**
   - ���SwanLab API��Կ�Ƿ���ȷ
   - ȷ����Ŀ�����Ƿ����
   - �鿴�ն˴�����Ϣ

3. **ʵ�鲻��ʾ**
   - ȷ��ʵ������û���ظ�
   - �����ĿȨ������
   - ˢ�������ҳ��

### ����ģʽ

������ϸ��־��
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### ����ģʽ

������粻�ȶ������Խ���SwanLab��
```bash
python start_swanlab_training.py --disable_swanlab
```

## ? ������Դ

- [SwanLab�ٷ��ĵ�](https://docs.swanlab.cn/)
- [SwanLab GitHub](https://github.com/SwanHubX/SwanLab)
- [Qwen2.5-VLģ���ĵ�](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## ? ����

��ӭ�ύIssue��Pull Request���Ľ�SwanLab���ɹ��ܣ�
