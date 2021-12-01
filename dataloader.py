import os
import torch
import torch.nn.functional as F
from collections import defaultdict

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
    
    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        tensor += 1.0
        return tensor / tensor.max()

    def __init__(self, path=MOVIELENS_PATH):
        self.path = path
        self.users = torch.load(os.path.join(self.path, "users.pt"))
        self.items = torch.load(os.path.join(self.path, "items.pt"))
        self.ratings = torch.load(os.path.join(self.path, "ratings.pt"))
        self.timestamps = torch.load(os.path.join(self.path, "timestamps.pt"))

        self.idx = torch.stack((self.users, self.items), axis=0)
        self.numUsers, self.numItems = int(
            torch.max(self.users))+1, int(torch.max(self.items))+1
        self.rating_interactions = self.vec_to_sparse(
            self.users, self.items, self.ratings,
            size=(self.numUsers, self.numItems),
            load_full=False,
        )
        self.time_interactions = self.vec_to_sparse(
            self.users, self.items, self.timestamps,
            size=(self.numUsers, self.numItems),
            load_full=False,
        )

        # Init mappings
        self.user_item_map = defaultdict(list)
        self.user_rating_map = defaultdict(list)
        self.user_rating_togo_map = defaultdict(list)
        self.user_time_map = defaultdict(list)
        
        for i, user in enumerate(self.users):
            user = user.item()
            self.user_item_map[user].append(self.items[i].item())
            self.user_rating_map[user].append(self.ratings[i].item())
            self.user_time_map[user].append(self.timestamps[i].item())

        # Get unique userIDs & padding length
        self.uIDs = list(self.user_time_map.keys())
        self.padding_length = max([len(x)
                                  for x in self.user_item_map.values()])

        # Convert mappings of list to tensors
        for uid in self.uIDs:
            self.user_item_map[uid] = torch.tensor(self.user_item_map[uid])
            self.user_rating_map[uid] = self.normalize(
                torch.tensor(self.user_rating_map[uid]).float()
            )
            self.user_time_map[uid] = torch.tensor(self.user_time_map[uid])

        # Generate rating togo
        for uid, rating in self.user_rating_map.items():
            s = rating.sum()
            self.user_rating_togo_map[uid] = torch.zeros(self.padding_length+1)
            self.user_rating_togo_map[uid][0] = s
            for i, r in enumerate(rating):
                s -= r
                self.user_rating_togo_map[uid][i+1] = s


    def get_indices(self) -> torch.Tensor:
        return self.idx

    def get_ratings(self, load_full=False) -> torch.Tensor:
        if load_full:
            return self.rating_interactions.to_dense()
        return self.rating_interactions

    def get_timestamps(self, load_full=False) -> torch.Tensor:
        if load_full:
            return self.time_interactions.to_dense()
        return self.time_interactions

    def __len__(self) -> int:
        return self.numUsers

    def __getitem__(self, idx: int) -> torch.Tensor:
        uID = self.uIDs[idx]
        len_diff = self.padding_length - self.user_item_map[uID].shape[0]
        items = F.pad(
            self.user_item_map[uID], (0, len_diff), "constant", 0
        )
        ratings = F.pad(
            self.user_rating_map[uID], (0, len_diff), "constant", 0
        )
        ratings_togo = self.user_rating_togo_map[uID]
        time = F.pad(
            self.user_time_map[uID], (0, len_diff), "constant", 0
        )

        return uID, items, ratings, ratings_togo, time


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

    # return the user-item Interaction matrices (ratings/timestamps)
    # Dim: numUsers x numItems
    ratings = user_dataset.get_ratings(load_full=True)
    timestamps = user_dataset.get_timestamps(load_full=True)

    # ...or you can access the dataset by index (through the getter function)
    # uID: int, user id (unique)
    # items: torch.Tensor, movie ids (unique) which reviewed by the user, sorted by timestamp in ascending order
    # ratings: torch.Tensor, normalized ratings of the movies reviewed by the user, 
    #          sorted by timestamp in ascending order
    # ratings_togo: torch.Tensor, cumulative normalized ratings, 
    #               sorted by timestamp in ascending order
    # time: torch.Tensor, timestamps of the movies reviewed by the user, 
    #                     sorted by timestamp in ascending order
    # items/ratings/times are padded with 0 to match the length of the largest user
    uID, items, ratings, ratings_togo, time = user_dataset[0]

    """
    3. Import Dataset object into a torch Dataloader
    """
    from torch.utils.data import DataLoader

    # In order to get data in batches, we can import our dataset into a torch dataloader
    train_loader = DataLoader(
        user_dataset, batch_size=32, shuffle=True, num_workers=16)

    # Here is an example of how you can use the dataloader
    for i, batch in enumerate(train_loader):
        if i < 3:
            uID, items, ratings, ratings_togo, time = batch
            print(f"User ID: {uID}")
            print(
                f"user_movies: {items.size()}, # movies: {(items>=0).sum()}")
            print(items)
            print()
            print(
                f"user_ratings: {ratings.size()}, # ratings: {(ratings>=0).sum()}")
            print(ratings)
            print()
            print(
                f"user_ratings_togo: {ratings_togo.size()}, # ratings_togo: {(ratings_togo>=0).sum()}")
            print(ratings_togo)
            print()
            print(
                f"user_timestamps: {time.size()}, # timestamps: {(time>=0).sum()}")
            print(time)
            print()
        else:
            break
