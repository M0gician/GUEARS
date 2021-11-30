import os
import torch

MOVIELENS_PATH = "ml-100k/"


class UserDataset(torch.utils.data.Dataset):
    """Implements User Dataloader"""

    @staticmethod
    def vec_to_sparse(i: torch.tensor, j: torch.tensor, v: torch.tensor,
                      size=None, load_full=False) -> torch.Tensor:
        idx = torch.stack((i, j), axis=0)
        sparsed_matrix = torch.sparse_coo_tensor(idx, v, size=size)
        if load_full:
            sparsed_matrix = sparsed_matrix.to_dense()
        return sparsed_matrix

    def __init__(self, path=MOVIELENS_PATH):
        self.path = path
        self.users = torch.load(os.path.join(self.path, "users.pt"))
        self.items = torch.load(os.path.join(self.path, "items.pt"))
        self.ratings = torch.load(os.path.join(self.path, "ratings.pt"))

        self.idx = torch.stack((self.users, self.items), axis=0)
        self.numUsers, self.numItems = int(
            torch.max(self.users))+1, int(torch.max(self.items))+1
        self.interactions = self.vec_to_sparse(
            self.users, self.items, self.ratings,
            size=(self.numUsers, self.numItems),
            load_full=False,
        )

    def get_indices(self) -> torch.Tensor:
        return self.idx

    def get_interactions(self, load_full=False) -> torch.Tensor:
        if load_full:
            return self.interactions.to_dense()
        return self.interactions

    def __len__(self) -> int:
        return self.numUsers

    def __getitem__(self, idx: int) -> torch.Tensor:
        user_ratings = self.interactions[idx].to_dense()

        return user_ratings, torch.tensor(idx)


if __name__ == '__main__':
    """
    0. Flags
        load_full: bool
        if `load_full` is True, any function which takes this flag will return a full matrix
        otherwise, it will return in the form of a spared matrix
    """

    """
    1. Load the unmodified dataset
    """
    user_dataset = UserDataset()

    """
    2. Access attributes within the Dataset object
    """
    # return indices for all valid entries within the dataset
    # returned value will be a 2d-matrix,
    # with 1st row as row indices and 2nd row as col indices
    indices = user_dataset.get_indices()

    # return the user-item Interactions
    interactions = user_dataset.get_interactions(load_full=True)

    # ...or you can access the dataset by index (through the getter function)
    user_ratings, idx = user_dataset[0]

    """
    3. Import Dataset object into a torch Dataloader
    """
    from torch.utils.data import DataLoader

    # In order to get data in batches, we can import our dataset into a torch dataloader
    train_loader = DataLoader(
        user_dataset, batch_size=1, shuffle=True, num_workers=16)

    # Here is an example of how you can use the dataloader
    for i, batch in enumerate(train_loader):
        if i < 3:
            user_ratings, idx = batch
            print(f"Internal User ID: {idx[0]}")
            print(
                f"user_ratings: {user_ratings.size()}, # raings: {(user_ratings>0).sum()}")
            print(user_ratings)
            print()
        else:
            break
