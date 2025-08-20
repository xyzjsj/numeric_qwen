#!/usr/bin/env python3
"""
�������ݴ�������
"""

import os
import json
import torch
from PIL import Image
import sys

# ��ӵ�ǰĿ¼��path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from numeric_qwen2_5_vl import NumericQwen2_5_VLProcessor
from transformers import Qwen2_5_VLProcessor as OriginalProcessor

def debug_data_processing():
    """
    �������ݴ������
    """
    print("=== �������ݴ��� ===")
    
    # ����ԭʼ�����������ǵĴ�����
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    try:
        original_processor = OriginalProcessor.from_pretrained(model_path)
        our_processor = NumericQwen2_5_VLProcessor.from_pretrained(model_path)
        
        print("? ���������سɹ�")
        
        # ��������
        data_path = "/data1/wangzhiye/1a1a11/original/data/numeric_training_data.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ���Ե�һ����ͼ�������
        image_sample = None
        for item in data[:10]:
            if 'image' in item:
                image_sample = item
                break
        
        if image_sample is None:
            print("? û���ҵ�����ͼ�������")
            return
            
        print(f"��������: {image_sample['id']}")
        print(f"ͼ��: {image_sample.get('image', 'None')}")
        
        # �����Ի��ı�
        conversations = image_sample.get('conversations', [])
        full_text = ""
        for turn in conversations:
            role = turn.get('from', '')
            content = turn.get('value', '')
            if role == 'human':
                full_text += f"Human: {content}\n"
            elif role == 'gpt':
                full_text += f"Assistant: {content}\n"
        
        print(f"�����ı�:\n{full_text}")
        
        # ����ͼ��
        image_path = os.path.join("/data1/wangzhiye/1a1a11/original/data/images", image_sample['image'])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            print(f"? ͼ����سɹ�: {image.size}")
        else:
            print(f"? ͼ���ļ�������: {image_path}")
            return
        
        # ����ԭʼ������
        print("\n=== ����ԭʼ������ ===")
        try:
            original_result = original_processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            print("? ԭʼ�������ɹ�")
            print(f"Keys: {list(original_result.keys())}")
            for key, value in original_result.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
                    
            # ����Ƿ���image_embeds
            if 'image_embeds' in original_result:
                print(f"  image_embeds shape: {original_result['image_embeds'].shape}")
            
            # ���input_ids�е�ͼ��tokens
            input_ids = original_result['input_ids']
            vision_start_token_id = getattr(original_processor.tokenizer, 'vision_start_token_id', 151652)
            vision_end_token_id = getattr(original_processor.tokenizer, 'vision_end_token_id', 151653)
            vision_token_id = getattr(original_processor.tokenizer, 'vision_token_id', 151654)
            
            vision_start_count = (input_ids == vision_start_token_id).sum().item()
            vision_end_count = (input_ids == vision_end_token_id).sum().item()  
            vision_token_count = (input_ids == vision_token_id).sum().item()
            
            print(f"  Vision start tokens: {vision_start_count}")
            print(f"  Vision end tokens: {vision_end_count}")
            print(f"  Vision tokens: {vision_token_count}")
            
        except Exception as e:
            print(f"? ԭʼ������ʧ��: {e}")
            import traceback
            traceback.print_exc()
        
        # �������ǵĴ�����
        print("\n=== �������ǵĴ����� ===")
        try:
            our_result = our_processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            print("? ���ǵĴ������ɹ�")
            print(f"Keys: {list(our_result.keys())}")
            for key, value in our_result.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
                    
        except Exception as e:
            print(f"? ���ǵĴ�����ʧ��: {e}")
            import traceback
            traceback.print_exc()
        
        # ����ֻ���ı������
        print("\n=== ����ֻ���ı������ ===")
        text_only = "���� 1 + 2 �Ľ���� <num><3>"
        try:
            text_result = our_processor(
                text=text_only,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            print("? ���ı�����ɹ�")
            print(f"Keys: {list(text_result.keys())}")
            for key, value in text_result.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
                    
        except Exception as e:
            print(f"? ���ı�����ʧ��: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"? �������ʧ��: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_processing()
