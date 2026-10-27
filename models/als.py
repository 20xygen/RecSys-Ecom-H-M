
import numpy as np
from scipy.linalg import solve
from tqdm.auto import tqdm


class ImplicitALS:
    def __init__(self, rank=32, alpha=10, regularization=0.1, iterations=5, seed=42):
        self.rank = rank
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def _update(self, X, fixed, targets=None):  # targets - чеховское ружье
        gram = fixed.T @ fixed
        eye = np.eye(self.rank, dtype=np.float32) * self.regularization
        updated = np.zeros((X.shape[0], self.rank), dtype=np.float32)

        for u in range(X.shape[0]):
            start, end = X.indptr[u:u + 2]
            if start == end:
                continue

            indices = X.indices[start:end]
            confidence = 1 + self.alpha * X.data[start:end]
            vectors = fixed[indices]

            A = gram + (vectors.T * (confidence - 1)) @ vectors + eye
            values = 1 if targets is None else targets.data[start:end]
            b = ((confidence * values)[:, None] * vectors).sum(axis=0)
            updated[u] = solve(A, b, assume_a="pos", check_finite=False)

        return updated

    def fit(self, X, popular_ids, verbose=False, targets=None):
        self.X = X
        self.popular_ids = np.asarray(popular_ids, dtype=np.int64)

        rng = np.random.default_rng(self.seed)
        self.user_embeds = rng.normal(0, 0.01, (X.shape[0], self.rank)).astype(np.float32)
        self.item_embeds = rng.normal(0, 0.01, (X.shape[1], self.rank)).astype(np.float32)

        X_T = X.T.tocsr()
        targets_T = targets.T.tocsr() if targets is not None else None
        
        for _ in tqdm(range(self.iterations), desc="ALS", disable=not verbose):
            self.user_embeds = self._update(X, self.item_embeds, targets)
            self.item_embeds = self._update(X_T, self.user_embeds, targets_T)

        return self

    def forward(self, user_ids, k=10):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        result = np.tile(self.popular_ids[:k], (len(user_ids), 1))

        valid = (user_ids >= 0) & (user_ids < self.X.shape[0])
        warm = np.zeros(len(user_ids), dtype=bool)
        warm[valid] = self.X.indptr[user_ids[valid]] < self.X.indptr[user_ids[valid] + 1]

        if not warm.any():
            return result

        scores = self.user_embeds[user_ids[warm]] @ self.item_embeds.T
        indices = np.argpartition(scores, -k, axis=1)[:, -k:]
        order = np.argsort(-np.take_along_axis(scores, indices, axis=1), axis=1)

        result[warm] = np.take_along_axis(indices, order, axis=1)
        return result
