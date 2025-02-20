import json

def cut_difficulties():
    # Process training data
    input_train = "./imagenet_256/train.jsonl"
    output_train = "./imagenet_256/train_cut.jsonl"
    
    with open(input_train, "r") as in_f, open(output_train, "w") as out_f:
        for line in in_f:
            data = json.loads(line)
            # Keep only tokens field
            cut_data = {"tokens": data["tokens"]}
            json.dump(cut_data, out_f)
            out_f.write("\n")
    
    print(f"Processed {input_train} -> {output_train}")

    # Process test data 
    input_test = "./imagenet_256/test.jsonl"
    output_test = "./imagenet_256/test_cut.jsonl"
    
    with open(input_test, "r") as in_f, open(output_test, "w") as out_f:
        for line in in_f:
            data = json.loads(line)
            # Keep only tokens field
            cut_data = {"tokens": data["tokens"]}
            json.dump(cut_data, out_f)
            out_f.write("\n")
            
    print(f"Processed {input_test} -> {output_test}")

if __name__ == "__main__":
    cut_difficulties()
