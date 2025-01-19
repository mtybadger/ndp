# Load and check training labels
with open('./imagenet/labels_train.txt', 'r') as f:
    labels = [int(line.strip()) for line in f.readlines()]

unique_labels = sorted(set(labels))
print(f"Number of labels: {len(labels)}")
print(f"Number of unique labels: {len(unique_labels)}")
print(f"Label range: {min(labels)} to {max(labels)}")

# Check if all labels between min and max exist
expected_labels = set(range(min(labels), max(labels) + 1))
missing_labels = expected_labels - set(labels)

if missing_labels:
    print("Missing labels:", sorted(missing_labels))
else:
    print("All labels in range are present")

# Print label distribution
from collections import Counter
label_counts = Counter(labels)
print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    print(f"Label {label}: {count} instances")
