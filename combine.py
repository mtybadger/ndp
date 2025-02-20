import os

from data.custom import CustomTrain

def combine_train_files():
    # Load the number of samples in the train dataset. 
    # (We rely on the environment having CustomTrain imported, as we cannot import here.)
    train_dataset = CustomTrain(
        training_images_list_file="./imagenet_256/train.txt",
        size=256
    )
    total_samples = len(train_dataset)
    print(f"Total training samples: {total_samples}")

    # Combine rank files into a single file
    output_file = "./imagenet_256/train.jsonl"
    with open(output_file, "w") as out_f:
        for rank in range(24):
            input_file = f"./imagenet_256/data/train_rank{rank}.jsonl"
            try:
                with open(input_file, "r") as in_f:
                    for line in in_f:
                        out_f.write(line)
            except FileNotFoundError:
                print(f"Could not find {input_file}, skipping.")
    
    print(f"Combined rank files into {output_file}")

if __name__ == "__main__":
    combine_train_files()
