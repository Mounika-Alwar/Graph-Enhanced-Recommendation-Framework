import json
import streamlit as st

from core.vector_builder import generate_all_embeddings
from core.recommender import recommend_users
from core.tag_recommender import recommend_tags
from core.cluster import show_user_cluster

@st.cache_data
def load_users():
    with open("data/users.json", "r") as f:
        return json.load(f)["users"]


st.set_page_config(page_title="Graph Enhanced Recommendation System", layout="wide")

users = load_users()
user_ids = list(users.keys())


st.title("Graph Enhanced Recommendation Framework")

st.subheader("1. Choose User")
selected_user = st.selectbox("Select a user", user_ids)


st.subheader("2. Build Unified Embedding Vector")
if st.button("Build Vector"):
    generate_all_embeddings()
    st.success("Unified embeddings built and stored successfully.")


st.subheader("3. Recommended Users")
if st.button("Show Recommended Users"):
    recs = recommend_users(selected_user, top_k=5)

    if not recs:
        st.warning("No recommendations found.")
    else:
        st.write("Top Recommended Users:")

        for rec in recs:
            with st.container():
                st.markdown(f"### {rec['user_id']}")
                st.write({
                    "Bio Similarity": round(rec["bio_similarity"], 4),
                    "Behaviour Similarity": round(rec["behaviour_similarity"], 4),
                    "Graph Similarity": round(rec["graph_similarity"], 4),
                    "Final Score": round(rec["final_score"], 4)
                })
                st.markdown("---")



st.subheader("4. Recommended Tags")
if st.button("Show Recommended Tags"):
    tag_recs = recommend_tags(selected_user, top_k=10)

    if not tag_recs:
        st.warning("No tag recommendations.")
    else:
        st.write("Top Tag Recommendations:")

        for rec in tag_recs:
            tag_name = rec["tag"]
            score_val = float(rec["score"])  # convert safely
            st.write(f"{tag_name} — score: {score_val:.3f}")
            st.markdown("---")





st.subheader("5. Show User Cluster")

if st.button("Show My Cluster"):
    st.write("### Cluster Result")

    # Capture the output of your existing function
    import io
    import sys

    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        show_user_cluster(selected_user, n_clusters=3)
    except ValueError as e:
        st.error(str(e))
    finally:
        sys.stdout = sys.__stdout__  # restore stdout

    output = buffer.getvalue()
    st.text(output)
