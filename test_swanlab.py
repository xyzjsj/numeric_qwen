#!/usr/bin/env python3
"""
测试SwanLab集成功能

验证SwanLab可视化是否正常工作
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import torch
import swanlab
import time
from datetime import datetime

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_config import (
    get_training_config,
    init_swanlab,
    log_to_swanlab,
    log_model_info_to_swanlab
)
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)


def test_swanlab_basic():
    """
    测试SwanLab基本功能
    """
    print("=== 测试SwanLab基本功能 ===")
    
    try:
        # 初始化SwanLab
        run = swanlab.init(
            project="qsinghua",
            experiment_name=f"test_basic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "test_type": "basic_functionality",
                "learning_rate": 0.001,
                "batch_size": 4,
                "model": "Qwen2.5-VL-3B"
            },
            description="测试SwanLab基本集成功能"
        )
        
        print("✅ SwanLab基本初始化成功")
        
        # 模拟一些训练指标
        for epoch in range(3):
            for step in range(5):
                global_step = epoch * 5 + step
                
                # 模拟损失下降
                loss = 2.0 * (0.9 ** global_step) + 0.1
                accuracy = min(0.95, 0.5 + global_step * 0.03)
                
                # 记录指标
                run.log({
                    "train/loss": loss,
                    "train/accuracy": accuracy,
                    "train/epoch": epoch,
                    "train/learning_rate": 0.001 * (0.95 ** epoch)
                }, step=global_step)
                
                print(f"  记录 epoch {epoch}, step {step}: loss={loss:.4f}, acc={accuracy:.4f}")
                time.sleep(0.1)  # 模拟训练时间
        
        # 记录最终结果
        run.log({
            "final/best_loss": 0.15,
            "final/best_accuracy": 0.92,
            "final/total_steps": 15
        })
        
        print("✅ 指标记录成功")
        
        # 完成实验
        run.finish()
        print("✅ SwanLab基本功能测试完成")
        print(f"🔗 查看结果: https://swanlab.cn/qsinghua")
        
        return True
        
    except Exception as e:
        print(f"❌ SwanLab基本功能测试失败: {e}")
        return False


def test_swanlab_integration():
    """
    测试与训练配置的集成
    """
    print("\n=== 测试SwanLab训练配置集成 ===")
    
    try:
        # 创建训练配置
        training_args = get_training_config(
            output_dir="/tmp/test_swanlab_output",
            model_path="Qwen/Qwen2.5-VL-3B-Instruct",
            data_path="/tmp/test_data.json",
            enable_swanlab=True,
            swanlab_project="qsinghua",
            swanlab_experiment=f"test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("✅ 训练配置创建成功")
        print(f"   SwanLab项目: {training_args.swanlab_project}")
        print(f"   SwanLab实验: {training_args.swanlab_experiment}")
        print(f"   启用SwanLab: {training_args.enable_swanlab}")
        
        # 初始化SwanLab
        swanlab_run = init_swanlab(training_args)
        
        if swanlab_run is not None:
            print("✅ SwanLab集成初始化成功")
            
            # 测试记录功能
            test_metrics = {
                "test/integration_status": 1.0,
                "test/config_loaded": 1.0,
                "model/type_numeric": 1.0  # 改为数值类型
            }
            
            log_to_swanlab(swanlab_run, test_metrics)
            print("✅ 集成记录功能测试成功")
            
            # 完成实验
            try:
                swanlab_run.finish()
            except:
                pass
                
            return True
        else:
            print("❌ SwanLab集成初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ SwanLab集成测试失败: {e}")
        return False


def test_swanlab_model_logging():
    """
    测试模型信息记录
    """
    print("\n=== 测试模型信息记录 ===")
    
    try:
        # 创建一个简单的模型配置进行测试
        from transformers import AutoConfig, AutoTokenizer
        
        # 模拟创建配置和tokenizer
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
        
        # 初始化SwanLab
        run = swanlab.init(
            project="qsinghua",
            experiment_name=f"test_model_logging_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={"test_type": "model_logging"},
            description="测试模型信息记录功能"
        )
        
        # 创建一个模拟的processor对象
        class MockProcessor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
        
        processor = MockProcessor(tokenizer)
        
        # 创建一个模拟的模型对象
        class MockModel:
            def __init__(self, config):
                self.config = config
                # 模拟一些参数
                self.dummy_param = torch.nn.Parameter(torch.randn(1000, 1000))
                
            def parameters(self):
                return [self.dummy_param]
                
            def num_parameters(self):
                return sum(p.numel() for p in self.parameters())
        
        model = MockModel(config)
        
        # 测试模型信息记录
        log_model_info_to_swanlab(run, model, processor)
        
        print("✅ 模型信息记录测试成功")
        
        # 完成实验
        run.finish()
        return True
        
    except Exception as e:
        print(f"❌ 模型信息记录测试失败: {e}")
        return False


def main():
    """
    运行所有测试
    """
    print("🚀 开始SwanLab集成测试")
    print("=" * 50)
    
    results = []
    
    # 测试1: 基本功能
    results.append(test_swanlab_basic())
    
    # 测试2: 集成功能
    results.append(test_swanlab_integration())
    
    # 测试3: 模型信息记录
    results.append(test_swanlab_model_logging())
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"   基本功能测试: {'✅ 通过' if results[0] else '❌ 失败'}")
    print(f"   集成功能测试: {'✅ 通过' if results[1] else '❌ 失败'}")
    print(f"   模型记录测试: {'✅ 通过' if results[2] else '❌ 失败'}")
    
    total_passed = sum(results)
    print(f"\n🎯 总体结果: {total_passed}/3 项测试通过")
    
    if total_passed == 3:
        print("🎉 所有测试通过！SwanLab集成准备就绪")
        print("🔗 查看所有实验: https://swanlab.cn/qsinghua")
    else:
        print("⚠️  部分测试失败，请检查SwanLab配置")
    
    return total_passed == 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
