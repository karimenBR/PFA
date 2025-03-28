import torch


def cuda_test():
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x @ y
    print(f"Matrix multiplication result (should be on GPU):\n{z}")
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")

if torch.cuda.is_available():
    cuda_test()
else:

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA device found")
    print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")