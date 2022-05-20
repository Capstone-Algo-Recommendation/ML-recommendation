import numpy as np

def recall_at_k(X_pred, heldout, k=30):
    tp = len(set(X_pred[:k]) & set(heldout))
    if len(heldout)>0:
      recall = tp/len(heldout)
      return recall
    else:
      return 0
  
def hit_rate_at_k(X_pred, heldout, k=30):
    tp = len(set(X_pred[:k]) & set(heldout))
    if tp >= 1:
      return 1
    else:
      return 0