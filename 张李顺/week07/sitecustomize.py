try:
    import torch
    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
