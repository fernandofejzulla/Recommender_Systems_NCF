import torch
import os
from data_utils import(load_movielens, convert_to_implicit, remap_ids, split_data, negative_sampling, create_dataset)
from model import NCF
from train import train_model
from evaluation import evaluate_model

#configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#load data
print("Loading data...")
df = load_movielens("../ml-1m/ratings.dat")
df = convert_to_implicit(df)
df, num_users, num_items = remap_ids(df)

print(f"Number of users: {num_users}, Number of items: {num_items}")

# Build the complete set of all positive pairs from the dataset before splitting,so that 
# no sampled negative accidentally be at the same time a positive interaction in another split.
all_positive_pairs = set(zip(df["user"], df["item"]))

#train/val/test
train_df, val_df, test_df = split_data(df)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

#negative sampling
print("Getting negative sampling...")

train_users, train_items, train_labels = negative_sampling(train_df, num_items, all_positive_pairs)
val_users, val_items, val_labels = negative_sampling(val_df, num_items, all_positive_pairs)

#pytorch datasets
train_dataset = create_dataset(train_users, train_items, train_labels)
val_dataset = create_dataset(val_users, val_items, val_labels)

#initializing 
print("initializing...")

mlp_layers=[64, 32, 16, 8]
model = NCF(num_users=num_users, num_items=num_items, embedding_dim=32, mlp_layers=mlp_layers)

#train
print("training...")

train_model(model, train_dataset, val_dataset, epochs=20, batch_size=256, learning_rate=0.001, 
            patience=3, device=device, mlp_layers=mlp_layers)

#load best model
model_filename = f"best_model_{mlp_layers}.pth"
if os.path.exists(model_filename):
    model.load_state_dict(torch.load(model_filename))
    model.to(device)
else:
    print("Warning: best_model.pth not found, using current model.")

#evaluate
print("evaluating...")

# pass train_df so evaluate_model can exclude items the user already interacted with 
# during training from the ranked list.
recall, ndcg = evaluate_model(model, test_df, train_df, num_items, device, k=10)

print("Recall@10:", recall)
print("NDCG@10:", ndcg)