
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


class FM(torch.nn.Module):
    def __init__(self, hidden_size, lr, regularization, epochs, batch_size, seed, device, user_columns, item_columns, optimizer, loss_fn):
        super().__init__()
        self.hidden_size = hidden_size
        self.lr = lr
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)
        self.user_columns = user_columns
        self.item_columns = item_columns
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @staticmethod
    def _encode(data, id_col, columns, offset):
        data = data.sort_values(id_col)
        features = [np.arange(len(data), dtype=np.int64) + offset + 1]
        offset += len(data)

        for col in columns:
            codes, values = pd.factorize(data[col], sort=True)
            features.append(codes + offset + 1)
            offset += len(values)

        return torch.tensor(np.column_stack(features), dtype=torch.long), offset

    def _parts(self, x):
        v = self.embeddings(x)
        v_sum = v.sum(dim=1)
        score = self.linear(x).sum(dim=(1, 2)) + 0.5 * (v_sum.square().sum(dim=1) - v.square().sum(dim=(1, 2)))
        return score, v_sum

    def _score(self, users, items):
        u_score, u_vectors = self._parts(self.user_features[users])
        i_score, i_vectors = self._parts(self.item_features[items])
        return self.bias + u_score + i_score + (u_vectors * i_vectors).sum(dim=1)

    def _negative_sampling(self, users, seen, rng):
        items = rng.integers(self.n_items, size=len(users))
        bad = seen[users, items].A1 > 0

        while bad.any():
            items[bad] = rng.integers(self.n_items, size=bad.sum())
            bad = seen[users, items].A1 > 0

        return items

    def fit(self, train, customers, articles, popular_ids, history=None, verbose=False):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.popular_ids = np.asarray(popular_ids, dtype=np.int64)
        self.n_users, self.n_items = len(customers), len(articles)

        users, offset = self._encode(customers, "user_id", self.user_columns, 0)
        items, offset = self._encode(articles, "item_id", self.item_columns, offset)

        positives = train[["user_id", "item_id"]].to_numpy(dtype=np.int64)
        users[~np.isin(np.arange(self.n_users), positives[:, 0]), 0] = 0

        self.user_features = torch.cat([users, torch.zeros((1, users.shape[1]), dtype=torch.long)]).to(self.device)
        self.item_features = items.to(self.device)

        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.linear = torch.nn.Embedding(offset + 1, 1, padding_idx=0)
        self.embeddings = torch.nn.Embedding(offset + 1, self.hidden_size, padding_idx=0)

        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.embeddings.weight, std=0.01)

        with torch.no_grad():
            self.embeddings.weight[0].zero_()

        self.to(self.device)
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.regularization)

        seen = csr_matrix((np.ones(len(positives)), (positives[:, 0], positives[:, 1])), shape=(self.n_users, self.n_items))

        progress = tqdm(range(self.epochs), desc="FM", disable=not verbose)

        for epoch in progress:
            self.train()
            order = rng.permutation(len(positives))
            total_loss = 0

            for start in range(0, len(order), self.batch_size):
                batch = positives[order[start:start + self.batch_size]]
                negative_items = self._negative_sampling(batch[:, 0], seen, rng)

                users = torch.tensor(np.tile(batch[:, 0], 2), device=self.device)
                items = torch.tensor(np.concatenate([batch[:, 1], negative_items]), device=self.device)
                targets = torch.tensor(np.r_[np.ones(len(batch)), np.zeros(len(batch))], dtype=torch.float32, device=self.device)

                loss = self.loss_fn(self._score(users, items), targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(targets)

            train_loss = total_loss / (2 * len(positives))

            if history is not None:
                history.append({"epoch": epoch + 1, "train_loss": train_loss})

            if verbose:
                progress.set_postfix(train_loss=round(train_loss, 4))

        return self

    @torch.no_grad()
    def forward(self, user_ids, k=10):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        safe_users = np.where((user_ids >= 0) & (user_ids < self.n_users), user_ids, self.n_users)

        self.eval()
        item_score, item_vectors = self._parts(self.item_features)
        user_score, user_vectors = self._parts(self.user_features[torch.tensor(safe_users, device=self.device)])
        scores = user_vectors @ item_vectors.T + user_score[:, None] + item_score[None, :] + self.bias

        return scores.topk(k, dim=1).indices.cpu().numpy()
