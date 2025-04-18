import torch

def verify_dl_environment():
    """深度学习环境完整性验证函数"""

    # 核心组件版本验证
    print("✅ PyTorch版本验证:", torch.__version__)
    print("✅ CUDA工具包版本:", torch.version.cuda)
    print("✅ cuDNN加速库版本:", torch.backends.cudnn.version())

    # GPU硬件信息检测
    if torch.cuda.is_available():
        print("\n🔥 GPU加速状态：已激活")
        print(f"• 当前显卡型号: {torch.cuda.get_device_name(0)}")
        print(f"• 可用GPU数量: {torch.cuda.device_count()}个")
        print(f"• 显存容量: {round(torch.cuda.get_device_properties(0).total_memory/1e9, 2)}GB")
    else:
        print("\n❌ GPU加速状态：未检测到可用显卡")

if __name__ == "__main__":
    verify_dl_environment()
