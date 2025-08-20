#!/usr/bin/env python3
"""
启动SwanLab可视化训练脚本

快速启动集成SwanLab的数值增强Qwen2.5-VL训练
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
from datetime import datetime

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="启动SwanLab可视化训练")
    parser.add_argument("--data_path", type=str, 
                       default="/data1/wangzhiye/1a1a11/original/data/numeric_training_data.json",
                       help="训练数据路径")
    parser.add_argument("--image_folder", type=str,
                       default="/data1/wangzhiye/1a1a11/original/data/images", 
                       help="图像文件夹路径")
    parser.add_argument("--output_dir", type=str,
                       default="/data1/wangzhiye/1a1a11/original/output",
                       help="输出目录")
    parser.add_argument("--model_path", type=str,
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--swanlab_project", type=str,
                       default="qsinghua",
                       help="SwanLab项目名称")
    parser.add_argument("--swanlab_experiment", type=str,
                       default=None,
                       help="SwanLab实验名称")
    parser.add_argument("--disable_swanlab", action="store_true",
                       help="禁用SwanLab可视化")
    parser.add_argument("--epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学习率")
    parser.add_argument("--test_only", action="store_true",
                       help="仅测试SwanLab集成，不进行实际训练")
    
    args = parser.parse_args()
    
    # 生成实验名称（如果未指定）
    if args.swanlab_experiment is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.swanlab_experiment = f"numeric_qwen2_5_vl_{timestamp}"
    
    print("🚀 启动SwanLab可视化训练")
    print("=" * 60)
    print("📋 训练配置:")
    print(f"   数据路径: {args.data_path}")
    print(f"   图像路径: {args.image_folder}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   基础模型: {args.model_path}")
    print(f"   训练轮数: {args.epochs}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   学习率: {args.learning_rate}")
    print()
    print("📊 SwanLab配置:")
    print(f"   项目名称: {args.swanlab_project}")
    print(f"   实验名称: {args.swanlab_experiment}")
    print(f"   启用状态: {'❌ 禁用' if args.disable_swanlab else '✅ 启用'}")
    if not args.disable_swanlab:
        print(f"   查看链接: https://swanlab.cn/{args.swanlab_project}")
    print("=" * 60)
    
    if args.test_only:
        print("🧪 运行SwanLab集成测试...")
        from test_swanlab import main as test_main
        success = test_main()
        if success:
            print("✅ SwanLab集成测试通过，可以开始训练")
        else:
            print("❌ SwanLab集成测试失败，请检查配置")
        return
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        print("请确保数据文件存在，或运行 --test_only 进行测试")
        return
    
    try:
        # 导入训练模块
        from training_config import get_training_config, init_swanlab
        from train import main as train_main
        
        # 设置环境变量
        os.environ["SWANLAB_PROJECT"] = args.swanlab_project
        os.environ["SWANLAB_EXPERIMENT"] = args.swanlab_experiment
        os.environ["ENABLE_SWANLAB"] = "false" if args.disable_swanlab else "true"
        os.environ["DATA_PATH"] = args.data_path
        os.environ["IMAGE_FOLDER"] = args.image_folder
        os.environ["OUTPUT_DIR"] = args.output_dir
        os.environ["MODEL_PATH"] = args.model_path
        os.environ["NUM_EPOCHS"] = str(args.epochs)
        os.environ["BATCH_SIZE"] = str(args.batch_size)
        os.environ["LEARNING_RATE"] = str(args.learning_rate)
        
        print("🎯 开始训练...")
        print()
        
        # 启动训练
        train_main()
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
