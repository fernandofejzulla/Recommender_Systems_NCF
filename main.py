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
df = load_movielens("../data/ml-1m/ratings.dat")
df = convert_to_implicit(df)
df, num_users, num_items = remap_ids(df)

print(f"Number of users: {num_users}, Number of items: {num_items}")

#train/val/test
train_df, val_df, test_df = split_data(df)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

#negative sampling
print("Getting negative sampling...")

train_users, train_items, train_labels = negative_sampling(train_df, num_items)
val_users, val_items, val_labels = negative_sampling(val_df, num_items)

#pytorch datasets
train_dataset = create_dataset(train_users, train_items, train_labels)
val_dataset = create_dataset(val_users, val_items, val_labels)

#initializing 
print("initializing...")

model = NCF(num_users=num_users, num_items=num_items, embedding_dim=32, mlp_layers=[128, 64, 32])

#train
print("training...")

train_model(model, train_dataset, val_dataset, epochs=20, batch_size=256, learning_rate=0.001, patience=3, device=device)

#load best model
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)
else:
    print("Warning: best_model.pth not found, using current model.")

#evaluate
print("evaluating...")

recall, ndcg = evaluate_model(model, test_df, num_items, device, k=10)

print("Recall@10:", recall)
print("NDCG@10:", ndcg)