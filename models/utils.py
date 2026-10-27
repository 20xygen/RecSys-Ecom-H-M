
import numpy as np
from scipy.sparse import csr_matrix


def interaction_matrix(data, n_users, n_items, binary=True):
    X = csr_matrix(
        (np.ones(len(data), dtype=np.float32), (data["user_id"], data["item_id"])),
        shape=(n_users, n_items),
    )

    if binary:
        X.data[:] = 1

    return X
