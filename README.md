# Transformer-Based Personalized Recommendation System

## 📌 Project Overview

This project develops a personalized recommendation system using **Transformer architectures and sequential user behavior modeling**.

The system learns patterns from users' historical interactions with movies and predicts items they may be interested in watching next. It uses the MovieLens dataset to train and evaluate recommendations, with an optional BERT-based text encoder for learning semantic representations of movie titles and descriptions.

The project combines recommendation systems, natural language processing, representation learning, and sequence modeling.

## 🎯 Objectives

* Build a personalized recommendation pipeline.
* Learn user preferences from historical interactions.
* Use Transformer models to capture sequential behavior patterns.
* Explore BERT embeddings for semantic item representations.
* Generate ranked recommendations for individual users.
* Evaluate recommendation quality using ranking metrics.

## 🧠 Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Hugging Face Transformers
* BERT
* MovieLens Dataset
* Scikit-learn

## 📂 Dataset

**MovieLens 1M**

MovieLens 1M contains approximately one million ratings from around 6,000 users across approximately 4,000 movies.

Dataset: https://grouplens.org/datasets/movielens/1m/

Although the initial implementation uses movie-rating data, the same general architecture can be adapted to video recommendation using video watch histories, titles, descriptions, categories, and engagement signals.

## ⚙️ System Architecture

1. **Data preprocessing:** Load movie metadata and user ratings.
2. **Interaction filtering:** Identify positive interactions using a defined rating threshold.
3. **Sequence generation:** Sort interactions chronologically and construct user histories.
4. **Embedding generation:** Map movie IDs to trainable embeddings.
5. **Transformer modeling:** Learn patterns in users' historical item sequences.
6. **Optional BERT integration:** Encode movie titles or descriptions into semantic vectors.
7. **Recommendation generation:** Rank candidate items and recommend the highest-scoring unseen items.
8. **Evaluation:** Measure ranking quality on held-out user interactions.

## 🔬 Key Features

* User interaction preprocessing
* Sequential recommendation modeling
* Transformer-based sequence learning
* Trainable movie embeddings
* Optional BERT-based semantic embeddings
* Personalized top-K recommendations
* Temporal train/test splitting
* Ranking-based model evaluation

## 🚀 Getting Started

### Install dependencies

```bash
pip install torch pandas numpy transformers scikit-learn
```

### Prepare the dataset

Download MovieLens 1M and extract its files into the project directory.

Expected structure:

```text
ml-1m/
├── movies.dat
├── ratings.dat
└── users.dat
```

### Run the project

```bash
python main.py
```

Configure the data paths and model settings in the script or configuration file before running.

## 📊 Evaluation Metrics

* Precision@K
* Recall@K
* NDCG@K
* Hit Rate@K
* Recall on held-out future interactions
* Training loss

A temporal split should be used where possible so the model predicts future interactions from past behavior rather than learning from randomly mixed histories.

## 🧩 Model Components

### Movie Embeddings

Represent movie IDs as trainable vectors that capture useful relationships between items.

### Transformer Encoder

Learn contextual relationships across a user's historical interaction sequence.

### BERT Text Encoder — Optional Extension

Generate semantic representations from movie titles or descriptions to complement collaborative behavior signals.

### Recommendation Layer

Produce scores for candidate movies and rank them for each user.

## 🔮 Future Improvements

* Integrate richer video metadata.
* Combine collaborative and content-based recommendation.
* Add watch time, completion rate, and engagement features.
* Compare the Transformer with matrix factorization and simpler sequential baselines.
* Implement candidate filtering and ranking.
* Deploy an interactive recommendation demo using Streamlit or FastAPI.

## 👨‍💻 Author

Developed as an AI/ML project exploring personalized recommendations, sequential modeling, and Transformer-based learning.
