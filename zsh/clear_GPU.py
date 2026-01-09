import torch
import gc

def clean_gpu():
    """
    清理所有可见 GPU 的显存：释放缓存并触发垃圾回收。
    """
    gc.collect()

    if torch.cuda.is_available():
        # 获取当前可见的 GPU 数量
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            with torch.cuda.device(i):  # 切换到第 i 个 GPU
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"GPU {i} cleaned. Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("CUDA is not available. No GPU to clean.")

if __name__ == "__main__":
    clean_gpu()