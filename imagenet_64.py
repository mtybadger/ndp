import datasets
import albumentations
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
# Define minimum dimensions required for cropping
MIN_SIZE = 64

# Define the transforms using Albumentations
transform = albumentations.Compose([
    # albumentations.SmallestMaxSize(max_size=MIN_SIZE, interpolation=2),  # interpolation=2 is cubic
    # albumentations.RandomCrop(height=MIN_SIZE, width=MIN_SIZE),
    albumentations.HorizontalFlip(p=0.5)
])

# Create output directories if they don't exist
os.makedirs("./imagenet_64/test", exist_ok=True)
os.makedirs("./imagenet_64/train", exist_ok=True)

def process_image(example, idx, split="test"):
    try:
        # Convert PIL Image to numpy array for Albumentations
        img = np.array(example['image'])
        
        # Apply transforms - this will automatically resize if too small
        transformed = transform(image=img)
        transformed_image = transformed['image']
        
        # Convert back to PIL for saving
        transformed_image = Image.fromarray(transformed_image)
        
        # Save the transformed image
        save_path = os.path.join(f"./imagenet_64/{split}", f"image_{idx}.jpg")
        transformed_image.save(save_path, quality=95)
        
    except Exception as e:
        print(f"Failed to process image {idx} in {split} split: {str(e)}")
        
    return example

def create_auxiliary_files(dataset):
    # Create train.txt from actual files in directory
    train_files = [f for f in os.listdir("./imagenet_64/train") if f.endswith('.jpg')]
    with open("./imagenet_64/train.txt", "w") as f:
        for filename in train_files:
            f.write(os.path.abspath(os.path.join("./imagenet_64/train", filename)) + "\n")
    
    # Create test.txt from actual files in directory
    test_files = [f for f in os.listdir("./imagenet_64/test") if f.endswith('.jpg')]
    with open("./imagenet_64/test.txt", "w") as f:
        for filename in test_files:
            f.write(os.path.abspath(os.path.join("./imagenet_64/test", filename)) + "\n")

    # Create labels.txt by matching image IDs to dataset labels
    # Get all labels in memory first    
    train_labels = dataset['train']['label']
    test_labels = dataset['test']['label']
    
    with open("./imagenet_64/labels_train.txt", "w") as f:
        for filename in tqdm(train_files):
            # Extract index from filename (e.g. "image_42.jpg" -> 42)
            idx = int(filename.split('_')[1].split('.')[0])
            f.write(f"{train_labels[idx]}\n")
            
    with open("./imagenet_64/labels_test.txt", "w") as f:
        for filename in tqdm(test_files):
            # Extract index from filename (e.g. "image_42.jpg" -> 42) 
            idx = int(filename.split('_')[1].split('.')[0])
            f.write(f"{test_labels[idx]}\n")

# Step 1: Load and process dataset
print("Step 1: Downloading and processing images...")
imagenet = datasets.load_dataset("benjamin-paine/imagenet-1k-64x64", trust_remote_code=True, cache_dir="./.cache")

# Process images in parallel using datasets.map
imagenet['train'] = imagenet['train'].map(
    lambda x, idx: process_image(x, idx, "train"),
    with_indices=True,
    num_proc=16
)

imagenet['test'] = imagenet['test'].map(
    lambda x, idx: process_image(x, idx, "test"),
    with_indices=True,
    num_proc=16
)

# Step 2: Create auxiliary files
print("Step 2: Creating auxiliary files...")
create_auxiliary_files(imagenet)

print("Done!")
