import pandas as pd


# =====================================
# Load movies
# =====================================

movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=[
        "movie_id",
        "title",
        "genres"
    ]
)


# =====================================
# Load ratings
# =====================================

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]
)


print(movies.head())

print(ratings.head())
#******************************
# Ratings
#******************************
positive_ratings = ratings[
    ratings["rating"] >= 4
].copy()

print(
    "Positive interactions:",
    len(positive_ratings)
)

#*******************************
#   User Behaving Sequences
#*******************************

user_sequences = (
    positive_ratings
    .sort_values("timestamp")
    .groupby("user_id")["movie_id"]
    .apply(list)
)


for user_id, sequence in list(
    user_sequences.items()
)[:5]:

    print(
        "User:",
        user_id
    )

    print(
        "History:",
        sequence[:10]
    )

#*********************************
#   Transformer Model
#*********************************

import torch
import torch.nn as nn


class VideoTransformer(nn.Module):

    def __init__(
        self,
        num_movies,
        embedding_dim=128,
        num_heads=4,
        num_layers=2
    ):

        super().__init__()


        # Movie embeddings
        self.movie_embedding = nn.Embedding(
            num_movies + 1,
            embedding_dim
        )


        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        # Prediction layer
        self.output_layer = nn.Linear(
            embedding_dim,
            num_movies + 1
        )


    def forward(self, movie_ids):

        # Convert movie IDs into vectors
        x = self.movie_embedding(
            movie_ids
        )


        # Transformer
        x = self.transformer(x)


        # Use final item representation
        x = x[:, -1, :]


        # Predict next movie
        output = self.output_layer(x)


        return output

#******************************
# Training Sequence
#******************************

training_sequences = []


for user_id, sequence in user_sequences.items():

    if len(sequence) < 4:
        continue

    for i in range(
        3,
        len(sequence)
    ):

        input_sequence = sequence[
            i-3:i
        ]

        target_movie = sequence[i]

        training_sequences.append(
            (
                input_sequence,
                target_movie
            )
        )


print(
    "Training examples:",
    len(training_sequences)
)

#*******************************
#    Train Transformer
#*******************************

from torch.utils.data import DataLoader


class RecommendationDataset:

    def __init__(self, sequences):

        self.sequences = sequences

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, index):

        sequence, target = self.sequences[index]

        return (
            torch.tensor(
                sequence,
                dtype=torch.long
            ),
            torch.tensor(
                target,
                dtype=torch.long
            )
        )


dataset = RecommendationDataset(
    training_sequences
)


loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

#*************************
#   Creating 
#*************************

num_movies = movies["movie_id"].max()

model = VideoTransformer(
    num_movies=num_movies,
    embedding_dim=128,
    num_heads=4,
    num_layers=2
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


criterion = nn.CrossEntropyLoss()

#****************************
# Training 
#****************************

for epoch in range(5):

    model.train()

    total_loss = 0

    for sequences, targets in loader:

        optimizer.zero_grad()


        outputs = model(
            sequences
        )


        loss = criterion(
            outputs,
            targets
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


    average_loss = (
        total_loss / len(loader)
    )


    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {average_loss:.4f}"
    )

#****************************
# Generate Recommendation
#****************************

def recommend_movies(
    model,
    user_history,
    movies,
    top_k=5
):

    model.eval()


    # Last 3 watched movies
    history = user_history[-3:]


    input_tensor = torch.tensor(
        [history],
        dtype=torch.long
    )


    with torch.no_grad():

        scores = model(
            input_tensor
        )


    # Get highest scores
    top_movies = torch.topk(
        scores,
        k=top_k
    ).indices[0].tolist()


    results = movies[
        movies["movie_id"].isin(
            top_movies
        )
    ]


    return results[
        ["movie_id", "title", "genres"]
    ]

user_id = 1

history = user_sequences[
    user_id
]


recommendations = recommend_movies(
    model,
    history,
    movies,
    top_k=5
)


print(
    recommendations
)


#*********************************
#     Adding BERT Embedding
#*********************************

from transformers import AutoTokenizer
from transformers import AutoModel

import torch


tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

bert = AutoModel.from_pretrained(
    "bert-base-uncased"
)


def get_text_embedding(text):

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )


    with torch.no_grad():

        output = bert(
            **tokens
        )


    embedding = output.last_hidden_state[
        :, 0, :
    ]


    return embedding

embedding = get_text_embedding(
    "The Dark Knight"
)

print(
    embedding.shape
)