import json
import numpy as np

# Utility

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def normalize_vector(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm



# Encoding functions

def encode_bio(bio):
    # Simple text-based embedding using hashing
    fields = [
        bio.get("profession", ""),
        " ".join(bio.get("interests", [])),
        " ".join(bio.get("tags_given", [])),
        " ".join(bio.get("tags_received", []))
    ]

    vector = []
    for field in fields:
        h = hash(field) % 1000
        vector.append(h / 1000)

    return normalize_vector(vector)


def encode_behaviour(beh):
    vector = [
        beh.get("logins_per_week", 0) / 10,
        beh.get("messages_sent", 0) / 50,
        beh.get("profile_views", 0) / 100,
        beh.get("time_spent_minutes", 0) / 500,
        len(beh.get("tags_given", [])) / 10,
        len(beh.get("tags_received", [])) / 10,
        len(beh.get("interactions", [])) / 10
    ]

    return normalize_vector(vector)


def encode_graph(user_id, graph, users):
    neighbours = graph.get("edges", [])
    tag_interactions = graph.get("tag_interactions", {})

    vector = [
        len(neighbours) / 10,               # connectivity score
        len(tag_interactions.keys()) / 10   # tag interaction richness
    ]

    return normalize_vector(vector)


# Unified Vector

def build_unified_vector(bio_vec, beh_vec, graph_vec):
    # Resize vectors to same length by padding
    max_len = max(len(bio_vec), len(beh_vec), len(graph_vec))

    def pad(v):
        return np.pad(v, (0, max_len - len(v)), mode='constant')

    b = pad(bio_vec)
    d = pad(beh_vec)
    g = pad(graph_vec)

    unified = 0.5 * b + 0.3 * d + 0.2 * g
    return normalize_vector(unified).tolist()



# Main update function

def generate_all_embeddings(user_json_path="data/users.json",
                            save_path="embeddings/unified_embeddings.json"):

    users = load_json(user_json_path)["users"]
    out = {}

    for uid, data in users.items():

        bio_vec = encode_bio(data["bio"])
        beh_vec = encode_behaviour(data["behaviour"])
        graph_vec = encode_graph(uid, data["graph"], users)

        unified = build_unified_vector(bio_vec, beh_vec, graph_vec)

        out[uid] = {
            "bio_vec": bio_vec.tolist(),
            "behaviour_vec": beh_vec.tolist(),
            "graph_vec": graph_vec.tolist(),
            "unified_vec": unified
        }

    save_json(save_path, out)
    return out


if __name__ == "__main__":
    generate_all_embeddings()
