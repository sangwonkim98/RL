import numpy as np
import random

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None

class GreedySimilarityPolicy:
    def __init__(self, max_docs, actual_feature_dim):
        self.max_docs = max_docs
        self.actual_feature_dim = actual_feature_dim
        
    def predict(self, obs, deterministic=True):
        # Unpack observation using the actual feature dimension
        q_feat = obs[:self.actual_feature_dim]
        doc_feats_flat = obs[self.actual_feature_dim : self.actual_feature_dim + self.max_docs * self.actual_feature_dim]
        doc_feats = doc_feats_flat.reshape(self.max_docs, self.actual_feature_dim)
        visited = obs[self.actual_feature_dim + (self.max_docs * self.actual_feature_dim) : -1]
        
        # Calculate cosine similarity (simplified as dot product since TF-IDF is normalized)
        similarities = np.dot(doc_feats, q_feat)
        
        # Mask visited docs
        masked_similarities = similarities.copy()
        masked_similarities[visited == 1] = -1.0
        
        best_doc = np.argmax(masked_similarities)
        
        # Stop if best similarity is too low or all docs visited
        if masked_similarities[best_doc] < 0.1 or np.all(visited == 1):
            return self.max_docs, None # STOP action
            
        return best_doc, None
