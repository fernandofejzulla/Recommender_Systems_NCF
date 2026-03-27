import torch
import numpy as np
from tqdm import tqdm

#recall@k function
def recal_k(recommended_items, ground_truth, k):
    recommended_items = recommended_items[:k]
    hits = len(set(recommended_items) & set(ground_truth))

    return hits / len(ground_truth)

#ndcg@k function
def ndcg_k(recommended_items, ground_truth, k):
    recommended_items = recommended_items[:k]

    dcg = 0

    for i, item in enumerate(recommended_items):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)

    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))

    return dcg / idcg if idcg > 0 else 0

#ranking evaluation
def evaluate_model(model, test_df, train_df, num_items, device, k=10):
    
    model.eval()
    recall_scores = []
    ndcg_scores = []

    train_items_per_user = train_df.groupby("user")["item"].apply(set).to_dict()

    users = test_df["user"].unique()

    with torch.no_grad():
        for user in tqdm(users):
            user_data = test_df[test_df["user"] == user]
            ground_truth = user_data["item"].tolist()

            items = np.arange(num_items)    #for each user we evaluate all items

            user_tensor = torch.tensor([user] * num_items).to(device)
            item_tensor = torch.tensor(items).to(device)

            scores = model(user_tensor, item_tensor)
            scores = scores.cpu().numpy()

            # Mask out training items by pushing their scores to -inf so they
            # are sorted to the bottom and never appear in the top-k results.
            train_items = train_items_per_user.get(user, set())
            for item_idx in train_items:
                scores[item_idx] = -np.inf

            ranked_items = np.argsort(scores)[::-1]

            recall = recal_k(ranked_items, ground_truth, k)
            ndcg = ndcg_k(ranked_items, ground_truth, k)

            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
    
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)

    return avg_recall, avg_ndcg