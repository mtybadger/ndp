import datasets
import albumentations
import numpy as np
from PIL import Image
import os

# Define minimum dimensions required for cropping
MIN_SIZE = 64

# Define the transforms using Albumentations
transform = albumentations.Compose([
    albumentations.SmallestMaxSize(max_size=MIN_SIZE, interpolation=2),  # interpolation=2 is cubic
    albumentations.RandomCrop(height=MIN_SIZE, width=MIN_SIZE),
    albumentations.HorizontalFlip(p=0.5)
])

# Create output directories if they don't exist
os.makedirs("./tinyimagenet/test", exist_ok=True)
os.makedirs("./tinyimagenet/train", exist_ok=True)

def process_image(example, idx, split="test"):
    # Convert PIL Image to numpy array for Albumentations
    img = np.array(example['image'])
    
    # Apply transforms - this will automatically resize if too small
    transformed = transform(image=img)
    transformed_image = transformed['image']
    
    # Convert back to PIL for saving
    transformed_image = Image.fromarray(transformed_image)
    
    # Save the transformed image
    save_path = os.path.join(f"./tinyimagenet/{split}", f"image_{idx}.jpg")
    transformed_image.save(save_path, quality=95)
    
    return example

def create_auxiliary_files(dataset):
    # Create train.txt
    with open("./tinyimagenet/train.txt", "w") as f:
        for idx in range(len(dataset['train'])):
            f.write(os.path.abspath(f"./tinyimagenet/train/image_{idx}.jpg\n"))
    
    # Create test.txt
    with open("./tinyimagenet/test.txt", "w") as f:
        for idx in range(len(dataset['valid'])):
            f.write(os.path.abspath(f"./tinyimagenet/test/image_{idx}.jpg\n"))
            
    # Create labels.txt
    with open("./tinyimagenet/labels.txt", "w") as f:
        for example in dataset['train']:
            f.write(f"{example['label']}\n")
        for example in dataset['valid']:
            f.write(f"{example['label']}\n")

# Step 1: Load and process dataset
print("Step 1: Downloading and processing images...")
imagenet = datasets.load_dataset("zh-plus/tiny-imagenet", trust_remote_code=True)

# # Process images in parallel using datasets.map
# imagenet['train'] = imagenet['train'].map(
#     lambda x, idx: process_image(x, idx, "train"),
#     with_indices=True,
# )

# imagenet['valid'] = imagenet['valid'].map(
#     lambda x, idx: process_image(x, idx, "test"),
#     with_indices=True,
# )

# Step 2: Create auxiliary files
print("Step 2: Creating auxiliary files...")
create_auxiliary_files(imagenet)

print("Done!")
