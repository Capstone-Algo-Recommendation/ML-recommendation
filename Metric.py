import numpy as np

def recall_at_k(X_pred, heldout, k=100):
    idx = np.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(X_pred.shape[0])[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout > 0)

    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall
  
def hit_rate_at_k(X_pred, heldout, k=100):
    idx = np.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(X_pred.shape[0])[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout > 0)

    tmp = np.logical_and(X_true_binary, X_pred_binary)
    hits = np.sum(tmp, axis=1)
    hits = np.count_nonzero(hits)
    return hits