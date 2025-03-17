from torch_fidelity import calculate_metrics
import numpy as np
import torch
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct, np.ndarray])

metrics_dict = calculate_metrics(
    input1="/mnt/ndp/imagenet_64/samples_16_ndp_cfg2.0",
    input2="/mnt/ndp/imagenet_64/train",
    cuda=True,
    cache_root="/mnt/ndp/fidelity", # Set custom cache directory,
    isc=True,
    fid=True,
    prc=True,
    input2_cache_name="imagenet64"
)

print(metrics_dict)