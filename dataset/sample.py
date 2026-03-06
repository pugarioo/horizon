from datasets import load_dataset

# Point to the specific subdirectory containing the split files
dataset = load_dataset("parquet", data_dir="dataset/gsm8k/main")

# Target the 'test' split key
sample = dataset["test"].shuffle(seed=42).select(range(298))

# Export to CSV
sample.to_csv("dataset/sampled_data.csv", index=True)

# Print verification
for index, row in enumerate(sample):
    print(f"Q: {row['question']}")
    print(f"A: {row['answer']}\n")
