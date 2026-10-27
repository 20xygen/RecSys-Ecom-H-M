
import numpy as np
from scipy.sparse import csr_matrix, diags


class I2I:
    def __init__(self, n_neighbors=100):
        self.n_neighbors = n_neighbors

    def fit(self, X, popular_ids):
        self.X = X  # CSR формат
        self.popular_ids = popular_ids

        counts = np.asarray(X.sum(axis=0)).ravel()  # у нас бинарное, так что .power(2) не нужно
        antinorm = np.zeros_like(counts)
        np.divide(1, np.sqrt(counts), out=antinorm, where=counts>0)

        similarity = (diags(antinorm) @ (X.T @ X) @ diags(antinorm)).tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        rows, cols, values = [], [], []

        n_items = X.shape[1]
        for i in range(n_items):
            start, end = similarity.indptr[i:i + 2]
            indices = similarity.indices[start:end]
            scores = similarity.data[start:end]

            if len(scores) > self.n_neighbors:
                selected = np.argpartition(scores, -self.n_neighbors)[-self.n_neighbors:]
                indices, scores = indices[selected], scores[selected]

            rows.extend([i] * len(indices))
            cols.extend(indices)
            values.extend(scores)

        self.similarity = csr_matrix(
            (values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32
        )
        return self

    def forward(self, user_ids, k=10):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        result = np.tile(self.popular_ids[:k], (len(user_ids), 1))
    
        valid = (user_ids >= 0) & (user_ids < self.X.shape[0])
        warm = np.zeros(len(user_ids), dtype=bool)
        warm[valid] = self.X.indptr[user_ids[valid]] < self.X.indptr[user_ids[valid] + 1]  # если строка пустая, значит данных по пользователю нет
    
        if not warm.any():
            return result
    
        scores = (self.X[user_ids[warm]] @ self.similarity).tocsr()

        for row, pos in enumerate(np.flatnonzero(warm)):
            start, end = scores.indptr[row:row + 2]
            indices = scores.indices[start:end]
            values = scores.data[start:end]
    
            positive = values > 0
            indices, values = indices[positive], values[positive]
    
            if len(values) > k:
                top = np.argpartition(values, -k)[-k:]
                indices, values = indices[top], values[top]
    
            selected = indices[np.argsort(-values)].tolist()
    
            if len(selected) < k:  # когда меньше k совпадений по покупкам с другими
                selected_set = set(selected)
                for item in self.popular_ids:
                    if item not in selected_set:
                        selected.append(int(item))
                        selected_set.add(int(item))
                        if len(selected) == k:
                            break
    
            result[pos] = selected[:k]
    
        return result
