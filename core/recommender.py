import json
import numpy as np

# Cosine Similarity

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Load Unified Embeddings
def load_embeddings(path="embeddings/unified_embeddings.json"):
    with open(path, "r") as f:
        return json.load(f)


# Recommend Users
def recommend_users(target_user_id, top_k=5):
    embeddings = load_embeddings()
    
    if target_user_id not in embeddings:
        raise ValueError("User not found in unified embeddings.")

    target = embeddings[target_user_id]

    results = []

    for uid, info in embeddings.items():
        if uid == target_user_id:
            continue

        # Extract vectors
        target_bio = np.array(target["bio_vec"])
        target_behav = np.array(target["behaviour_vec"])
        target_graph = np.array(target["graph_vec"])

        other_bio = np.array(info["bio_vec"])
        other_behav = np.array(info["behaviour_vec"])
        other_graph = np.array(info["graph_vec"])

        # Compute similarities
        bio_sim = cosine_sim(target_bio, other_bio)
        beh_sim = cosine_sim(target_behav, other_behav)
        graph_sim = cosine_sim(target_graph, other_graph)

        # Final weighted score
        final_score = (
            0.5 * bio_sim +
            0.3 * beh_sim +
            0.2 * graph_sim
        )

        results.append({
            "user_id": uid,
            "bio_similarity": round(bio_sim, 4),
            "behaviour_similarity": round(beh_sim, 4),
            "graph_similarity": round(graph_sim, 4),
            "final_score": round(final_score, 4)
        })

    # sort by best score
    ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return ranked[:top_k]

if __name__ == "__main__":
    print("\nTop recommendations for user_1:\n")
    recs = recommend_users("u101", top_k=3)
    for r in recs:
        print(r)
