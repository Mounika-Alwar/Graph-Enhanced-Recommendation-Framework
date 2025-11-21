import json
from collections import Counter


# load users from JSON

def load_users(path="data/users.json"):
    with open(path, "r") as f:
        data = json.load(f)
        return data["users"]   # your exact JSON structure



# Extract all tag-like signals from a user

def get_user_tags(user):
    tags = []

    # Bio interests (skills)
    tags.extend(user["bio"].get("interests", []))

    # Behaviour tags
    tags.extend(user["behaviour"].get("tags_given", []))
    tags.extend(user["behaviour"].get("tags_received", []))

    # Graph tags = tag interactions
    for connected_user, tlist in user["graph"].get("tag_interactions", {}).items():
        tags.extend(tlist)

    return list(set(tags))   # remove duplicates



# Tag Recommendation Engine

def recommend_tags(target_user_id, top_k=10):
    users = load_users()

    if target_user_id not in users:
        raise ValueError("User not found in users.json")

    target = users[target_user_id]
    target_tags = set(get_user_tags(target))

    tag_scores = Counter()

    # Compare with every other user
    for uid, user in users.items():
        if uid == target_user_id:
            continue

        
        # 1. BIO similarity
        
        bio_overlap = len(
            set(target["bio"]["interests"]) &
            set(user["bio"]["interests"])
        )

        
        # 2. BEHAVIOUR similarity
        
        behaviour_overlap = len(
            set(target["behaviour"]["interactions"]) &
            set(user["behaviour"]["interactions"])
        )

        # 3. TAG similarity
        tag_overlap = len(
            set(target["behaviour"]["tags_given"] + target["behaviour"]["tags_received"]) &
            set(user["behaviour"]["tags_given"] + user["behaviour"]["tags_received"])
        )

        
        # 3. GRAPH similarity
        
        graph_overlap = len(
            set(target["graph"]["edges"]) &
            set(user["graph"]["edges"])
        )

        # Weighted score
        similarity = (
            0.45 * bio_overlap +
            0.35 * behaviour_overlap +
            0.10 * tag_overlap +
            0.10 * graph_overlap
        )

        # Add tags with similarity weight
        if similarity > 0:
            for tag in get_user_tags(user):
                tag_scores[tag] += similarity

    # Remove tags the user already has
    for t in target_tags:
        tag_scores.pop(t, None)

    # Return top-K
    recommended = tag_scores.most_common(top_k)

    return [{"tag": tag, "score": round(score, 3)} for tag, score in recommended]



if __name__ == "__main__":
    print("\nTag Recommendations for u101:\n")
    recs = recommend_tags("u101", top_k=5)
    for r in recs:
        print(r)
