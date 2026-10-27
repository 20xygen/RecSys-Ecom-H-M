
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from models.als import ImplicitALS


class TemporalALS(ImplicitALS):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def fit(self, X, train, popular_ids, cutoff, verbose=False):
        days = (pd.Timestamp(cutoff) - train["t_dat"]).dt.days.to_numpy()
        weights = np.exp(-days / self.tau).astype(np.float32)
        recency = csr_matrix((weights, (train["user_id"], train["item_id"])), shape=X.shape)

        targets = X.copy()
        targets.data = -np.expm1(-recency.data)

        return super().fit(X, popular_ids, verbose=verbose, targets=targets)
