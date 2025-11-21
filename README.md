# Graph Enhanced Recommendation Framework

## Overview
The **Graph Enhanced Recommendation Framework** is a multi-perspective recommendation system that combines **user bio, behaviour, and graph/network interactions** to generate smarter user and tag recommendations. The system also allows clustering of users and evaluation through engagement metrics.

This framework is implemented in **Python** using **Streamlit** for the interactive demo, **NumPy** for vector operations, and **scikit-learn** for clustering.

---

## Features
1. **Unified Embeddings:**  
   Combine user bio, behaviour, and graph embeddings into a single vector for improved recommendation quality.
   
2. **User Recommendations:**  
   Recommend top users similar to a selected user using a weighted combination of bio, behaviour, and graph similarities.
   
3. **Tag Recommendations:**  
   Recommend relevant tags for a user based on similarity across bio, behaviour, and graph features.
   
4. **User Clustering:**  
   Cluster users based on unified embeddings and visualize users belonging to the same cluster.
   
5. **Metrics for Insights:**  
   Evaluate system effectiveness with metrics like:  
   - Time to first tag  
   - Tag diversity per user  
   - Reciprocity rate  
   - Recommendation CTR  
   - Embedding drift

---


