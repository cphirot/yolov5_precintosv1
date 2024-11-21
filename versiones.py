import torch

print("Versión de PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Versión de CUDA utilizada por PyTorch:", torch.version.cuda)
