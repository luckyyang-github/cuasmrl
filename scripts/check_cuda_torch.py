#!/usr/bin/env python3
import torch
import sys

def main():
    print("=" * 40)
    print(" PyTorch CUDA 环境检测")
    print("=" * 40)

    # PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")

    # 是否支持 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")

    # PyTorch 编译时的 CUDA 版本
    print(f"PyTorch 编译时使用的 CUDA 版本: {torch.version.cuda}")

    if cuda_available:
        # 当前 GPU 数量
        device_count = torch.cuda.device_count()
        print(f"检测到 GPU 数量: {device_count}")

        # 列出所有 GPU
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    设备能力: {torch.cuda.get_device_capability(i)}")
            print(f"    显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("未检测到可用的 CUDA GPU。可能原因: 驱动未安装 / PyTorch 是 CPU 版本 / 环境变量错误。")

    print("=" * 40)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("运行检测时出错:", e)
        sys.exit(1)
