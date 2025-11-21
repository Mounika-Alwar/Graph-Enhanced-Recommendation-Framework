import json
import numpy as np
from sklearn.cluster import KMeans


with open("data/users.json", "r") as f:
    DATA = json.load(f)["users"]


# Build vocabularies 

def build_vocabs(data):
    interest_vocab = set()
    tag_vocab = set()

    for uid, info in data.items():
        bio = info["bio"]
        behaviour = info["behaviour"]

        interest_vocab.update(bio.get("interests", []))
        tag_vocab.update(bio.get("tags_given", []))
        tag_vocab.update(bio.get("tags_received", []))
        tag_vocab.update(behaviour.get("tags_given", []))
        tag_vocab.update(behaviour.get("tags_received", []))

    return list(interest_vocab), list(tag_vocab)


INTEREST_VOCAB, TAG_VOCAB = build_vocabs(DATA)


# Make user vector

def user_to_vector(uid):
    user = DATA[uid]

    bio = user["bio"]
    behaviour = user["behaviour"]
    graph = user["graph"]

    vec = []

    # Interests one-hot
    interests = set(bio.get("interests", []))
    vec.extend([1 if i in interests else 0 for i in INTEREST_VOCAB])

    # Tag-based features
    tags_given = set(bio.get("tags_given", []) + behaviour.get("tags_given", []))
    tags_received = set(bio.get("tags_received", []) + behaviour.get("tags_received", []))

    vec.extend([1 if t in tags_given else 0 for t in TAG_VOCAB])
    vec.extend([1 if t in tags_received else 0 for t in TAG_VOCAB])

    # Behaviour numeric stats
    vec.append(behaviour.get("logins_per_week", 0))
    vec.append(behaviour.get("messages_sent", 0))
    vec.append(behaviour.get("profile_views", 0))
    vec.append(behaviour.get("time_spent_minutes", 0))

    # Graph features
    vec.append(len(graph.get("edges", [])))
    vec.append(sum(len(v) for v in graph.get("tag_interactions", {}).values()))

    return np.array(vec, dtype=float)


#Build Full Embedding Matrix

def build_matrix():
    X = []
    U = []

    for uid in DATA:
        U.append(uid)
        X.append(user_to_vector(uid))

    return U, np.vstack(X)


# Clustering
def cluster_users(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels


# Show user cluster
def show_user_cluster(uid, n_clusters=3):
    users, matrix = build_matrix()
    labels = cluster_users(matrix, n_clusters=n_clusters)

    target_idx = users.index(uid)
    target_label = labels[target_idx]

    print(f"\nUser {uid} belongs to cluster {target_label}\n")
    print("Users in same cluster:")
    for u, lbl in zip(users, labels):
        if lbl == target_label:
            print(" -", u)



if __name__ == "__main__":
    show_user_cluster("u104", n_clusters=3)

