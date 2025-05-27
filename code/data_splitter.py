import pandas as pd
import os

# Load and preprocess the dataset
df = pd.read_csv("./datasets/HeartDiseaseDataset.csv")  
df = df.astype("float64")
df["label"] = df["label"].astype("int64")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

# Standard normalization
df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()

# Split into equal parts (IID)
num_clients = 2
split_size = len(df) // num_clients

# Output folder for client datasets
output_folder = "datasets/client_splits"
os.makedirs(output_folder, exist_ok=True)

# Save each client dataset to a CSV file
for i in range(num_clients):
    start = i * split_size
    end = (i + 1) * split_size if i != num_clients - 1 else len(df)
    client_df = df.iloc[start:end]
    client_df.to_csv(os.path.join(output_folder, f"client_{i}.csv"), index=False)

print(f"✅ {num_clients} client datasets saved to '{output_folder}'")
