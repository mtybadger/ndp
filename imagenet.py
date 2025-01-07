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

# Create output directory if it doesn't exist
os.makedirs("/home/ubuntu/cifar-testing/ndp/tinyimagenet/test", exist_ok=True)

def process_image(example, idx):
    # Convert PIL Image to numpy array for Albumentations
    img = np.array(example['image'])
    
    # Apply transforms - this will automatically resize if too small
    transformed = transform(image=img)
    transformed_image = transformed['image']
    
    # Convert back to PIL for saving
    transformed_image = Image.fromarray(transformed_image)
    
    # Save the transformed image
    save_path = os.path.join("/home/ubuntu/cifar-testing/ndp/tinyimagenet/test", f"image_{idx}.jpg")
    transformed_image.save(save_path, quality=95)
    
    return example

# Load dataset
imagenet = datasets.load_dataset("zh-plus/tiny-imagenet", trust_remote_code=True)

# Process images in parallel using datasets.map
imagenet['valid'] = imagenet['valid'].map(
    process_image,
    with_indices=True,
)
