import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MovieLensDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, label_tensor):
        self.user = user_tensor
        self.item = item_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return {
            'user': self.user[idx],
            'item': self.item[idx],
            'label': self.labels[idx]
        }

def load_movielens(path): #ratings.dat
    df = pd.read_csv(path,
                     sep="::",
                     engine="python",
                     names=["user", "item", "rating", "timestamp"]
    )
    return df

def convert_to_implicit(df): #keep only positive interactions
    df = df[df["rating"] >= 4]
    df = df[["user", "item"]]

    return df

def remap_ids(df):
    user_map = {u: i for i, u in enumerate(df["user"].unique())}
    item_map = {i: j for j, i in enumerate(df["item"].unique())}

    df["user"] = df["user"].map(user_map)
    df["item"] = df["item"].map(item_map)

    num_users = len(user_map)
    num_items = len(item_map)

    return df, num_users, num_items

def split_data(df, train_ratio=0.7, val_ratio=0.15): #random train/val/test split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df

def negative_sampling(df, num_items, num_negatives=4): #sample negatives for each interaction
    user_item_set = set(zip(df["user"], df["item"]))

    users = []
    items = []
    labels = []

    for _, row in df.iterrows():
        u = row["user"]
        i = row["item"]
        #positive sample
        users.append(u)
        items.append(i)
        labels.append(1)

        #negative samples
        for _ in range(num_negatives):
            j = random.randint(0, num_items - 1)

            while (u, j) in user_item_set:
                j = random.randint(0, num_items - 1)
            
            users.append(u)
            items.append(j)
            labels.append(0)

    return users, items, labels

def create_dataset(users, items, labels):

    user_tensor = torch.tensor(users, dtype=torch.long)
    item_tensor = torch.tensor(items, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    dataset = MovieLensDataset(user_tensor, item_tensor, label_tensor)

    return dataset