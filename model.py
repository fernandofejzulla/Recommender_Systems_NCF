import torch
import torch.nn as nn 

class NCF(nn.Module): #neural collaborative filtering model
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        #GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        #MLP embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        #MLP layers
        mlp_input_dim = embedding_dim * 2
        layers = []

        for layer_size in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, layer_size))
            layers.append(nn.ReLU())
            mlp_input_dim = layer_size

        self.mlp = nn.Sequential(*layers)

        #prediction layer
        final_input_size = embedding_dim + mlp_layers[-1]
        self.output_layer = nn.Linear(final_input_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item): #forward pass 
        #GMF branch
        user_gmf = self.user_embedding_gmf(user)
        item_gmf = self.item_embedding_gmf(item)

        gmf_output = user_gmf * item_gmf

        #MLP branch
        user_mlp = self.user_embedding_mlp(user)
        item_mlp = self.item_embedding_mlp(item)

        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        #fusion
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        logits = self.output_layer(combined)
        prediction = self.sigmoid(logits)

        return prediction.squeeze()