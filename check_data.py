import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Read first 1% of lines from train.jsonl
with open('./imagenet_256/train.jsonl', 'r') as f:
    # Get total line count first
    total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines}")
    f.seek(0)
    
    # Initialize counters for each region
    region1_counts = defaultdict(int)  # 0-256
    region2_counts = defaultdict(int)  # 256-8448 (256+8192)
    region3_counts = defaultdict(int)  # 8448-12544 (8448+4096)
    region4_counts = defaultdict(int)  # 12544-16640 (12544+4096)
    # Process first 1% of lines
    sample_size = total_lines
    
    for i, line in tqdm(enumerate(f), total=sample_size, desc="Processing tokens"):
        if i >= sample_size:
            break
            
        data = json.loads(line)
        tokens = data['tokens']
        
        for token in tokens:
            if 1024 <= token < 1024 + 2048:
                region1_counts[token] += 1
            elif 1024 + 2048 <= token < 1024 + 2048 + 2048:
                region2_counts[token] += 1
            elif 1024 + 2048 + 2048 <= token < 1024 + 2048 + 2048 + 2048:
                region3_counts[token] += 1
            elif 1024 + 2048 + 2048 + 2048 <= token < 1024 + 2048 + 2048 + 2048 + 2048:
                region4_counts[token] += 1

    # Calculate total tokens in each region
    total_tokens = sum(region1_counts.values()) + sum(region2_counts.values()) + sum(region3_counts.values()) + sum(region4_counts.values())
    
    # Calculate and print percentages
    print("\nToken Usage Distribution:")
    print(f"Region 1 (0-256): {sum(region1_counts.values())/total_tokens*100:.2f}%")
    print(f"Region 2 (256-8448): {sum(region2_counts.values())/total_tokens*100:.2f}%")
    print(f"Region 3 (8448-12544): {sum(region3_counts.values())/total_tokens*100:.2f}%")
    print(f"Region 4 (12544-16640): {sum(region4_counts.values())/total_tokens*100:.2f}%")
    
    # Print codebook usage (what percentage of possible tokens in each region are actually used)
    print("\nCodebook Usage:")
    print(f"Region 1: {len(region1_counts)/2048*100:.2f}% of possible tokens used")
    print(f"Region 2: {len(region2_counts)/2048*100:.2f}% of possible tokens used")
    print(f"Region 3: {len(region3_counts)/2048*100:.2f}% of possible tokens used")
    print(f"Region 4: {len(region4_counts)/2048*100:.2f}% of possible tokens used")