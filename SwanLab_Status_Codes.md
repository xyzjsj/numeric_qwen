# SwanLab״̬��˵��

����SwanLabֻ֧����ֵ���͵�ͼ�����ݣ�����ʹ����ֵ״̬������ʾ��ͬ��ѵ��״̬��

## ѵ��״̬�� (training/status)

- **1.0**: ѵ����ʼ (Started)
- **2.0**: ѵ���ɹ���� (Completed Successfully)  
- **0.0**: ѵ�����ж� (Interrupted)
- **-1.0**: ѵ��ʧ�� (Failed)

## ����״̬�� (eval/status)

- **1.0**: ������ʼ (Evaluation Started)
- **2.0**: ������� (Evaluation Completed)
- **-1.0**: ����ʧ�� (Evaluation Failed)

## ����ֵ״̬��

- **1.0**: True/��/�ɹ�
- **0.0**: False/��/ʧ��

## ʾ��ָ��

```python
# ѵ����ʼ
{
    "training/status": 1.0,
    "training/resume_from_checkpoint": 1.0  # ����Ӽ���ָ�
}

# ѵ�����
{
    "training/status": 2.0,
    "training/model_saved": 1.0,
    "training/total_time": 3600.5,
    "training/total_steps": 1000
}

# ����״̬
{
    "training/status": -1.0,
    "training/error_occurred": 1.0
}
```

������Ƶĺô��ǣ�
1. ����״̬������SwanLab����ȷ��ʾΪͼ��
2. ���Ժ����׵���ͼ���п���ѵ���Ĳ�ͬ�׶�
3. ��ֵ״̬������ȷ�ĺ������������
