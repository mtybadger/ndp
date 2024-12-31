from data.custom import CustomTrain, CustomTest
from models.vqgan import VQMultiModel
from data.utils import custom_collate
import torch

model = VQMultiModel.load_from_checkpoint("/shared/imagenet/vqgan_multi_8_logs/last.ckpt")
model.to("cuda")
model.eval()

dataset = CustomTrain(
            training_images_list_file="/shared/imagenet/train.txt",
            size=256
        )

print(custom_collate([dataset[0]]))

with torch.no_grad():
    x = model.get_input(custom_collate([dataset[0]]), model.image_key)
    y = model(x.to("cuda"))
    
print(y)