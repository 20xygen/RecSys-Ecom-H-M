
import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


class SLIM:
    def __init__(self, n_candidates, l1, l2, iterations, precision):
        self.n_candidates = n_candidates
        self.l1 = l1
        self.l2 = l2
        self.iterations = iterations
        self.precision = precision

    def fit(self, X, popular_ids, verbose=False):
        self.X = X
        self.popular_ids = np.asarray(popular_ids, dtype=np.int64)

        X_csc = X.tocsc()
        gram = (X.T @ X).tocsc()
        gram.setdiag(0)
        gram.eliminate_zeros()

        n_users, n_items = X.shape
        counts = np.diff(X_csc.indptr)
        antinorm = np.zeros(n_items, dtype=np.float32)
        np.divide(1, np.sqrt(counts), out=antinorm, where=counts > 0)

        rows, cols, values = [], [], []

        for j in tqdm(range(n_items), desc="SLIM", disable=not verbose):
            start, end = gram.indptr[j:j + 2]
            neighbors = gram.indices[start:end]

            if len(neighbors) == 0:
                continue

            similarities = gram.data[start:end] * antinorm[neighbors] * antinorm[j]

            if len(neighbors) > self.n_candidates:
                top = np.argpartition(similarities, -self.n_candidates)[-self.n_candidates:]
                neighbors, similarities = neighbors[top], similarities[top]

            neighbors = neighbors[np.argsort(-similarities)]

            residual = np.zeros(n_users, dtype=np.float32)
            residual[X_csc.indices[X_csc.indptr[j]:X_csc.indptr[j + 1]]] = 1
            coefficients = np.zeros(len(neighbors), dtype=np.float32)

            for _ in range(self.iterations):
                max_change = 0

                for pos, i in enumerate(neighbors):
                    users = X_csc.indices[X_csc.indptr[i]:X_csc.indptr[i + 1]]
                    old = coefficients[pos]
                    new = max(0, (residual[users].sum() + counts[i] * old - self.l1) / (counts[i] + self.l2))
                    delta = new - old

                    if delta != 0:
                        residual[users] -= delta
                        coefficients[pos] = new
                        max_change = max(max_change, abs(delta))

                if max_change < self.precision:
                    break

            nonzero = coefficients > 0
            rows.extend(neighbors[nonzero].tolist())
            cols.extend([j] * int(nonzero.sum()))
            values.extend(coefficients[nonzero].tolist())

        self.W = csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        return self

    def forward(self, user_ids, k=10):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        result = np.tile(self.popular_ids[:k], (len(user_ids), 1))

        valid = (user_ids >= 0) & (user_ids < self.X.shape[0])
        warm = np.zeros(len(user_ids), dtype=bool)
        warm[valid] = self.X.indptr[user_ids[valid]] < self.X.indptr[user_ids[valid] + 1]

        if not warm.any():
            return result

        scores = (self.X[user_ids[warm]] @ self.W).tocsr()

        for row, pos in enumerate(np.flatnonzero(warm)):
            start, end = scores.indptr[row:row + 2]
            indices, values = scores.indices[start:end], scores.data[start:end]

            positive = values > 0
            indices, values = indices[positive], values[positive]

            if len(values) > k:
                top = np.argpartition(values, -k)[-k:]
                indices, values = indices[top], values[top]

            selected = indices[np.argsort(-values)].tolist()

            if len(selected) < k:
                selected_set = set(selected)
                for item in self.popular_ids:
                    if item not in selected_set:
                        selected.append(int(item))
                        selected_set.add(int(item))
                        if len(selected) == k:
                            break

            result[pos] = selected[:k]

        return result
