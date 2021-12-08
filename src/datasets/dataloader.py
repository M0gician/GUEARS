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

    def __init__(self, path=MOVIELENS_PATH, mode='train'):
        self.path = path
        if mode == 'train':
            self.users = torch.load(os.path.join(self.path, "trainusers.pt"))
            self.items = torch.load(os.path.join(self.path, "trainitems.pt"))
            self.ratings = torch.load(os.path.join(self.path, "trainratings.pt"))
            self.timestamps = torch.load(os.path.join(self.path, "traintimestamps.pt"))

            self.test_users = torch.load(os.path.join(self.path, "valusers.pt"))
            self.test_items = torch.load(os.path.join(self.path, "valitems.pt"))
            self.test_ratings = torch.load(os.path.join(self.path, "valratings.pt"))
            self.test_timestamps = torch.load(os.path.join(self.path, "valtimestamps.pt"))
        elif mode == 'test':
            self.users = torch.load(os.path.join(self.path, "b_trainusers.pt"))
            self.items = torch.load(os.path.join(self.path, "b_trainitems.pt"))
            self.ratings = torch.load(os.path.join(self.path, "b_trainratings.pt"))
            self.timestamps = torch.load(os.path.join(self.path, "b_traintimestamps.pt"))

            self.test_users = torch.load(os.path.join(self.path, "b_testusers.pt"))
            self.test_items = torch.load(os.path.join(self.path, "b_testitems.pt"))
            self.test_ratings = torch.load(os.path.join(self.path, "b_testratings.pt"))
            self.test_timestamps = torch.load(os.path.join(self.path, "b_testtimestamps.pt"))

        self.idx = torch.stack((self.users, self.items), axis=0)
        self.numUsers, self.numItems = int(
            torch.max(self.users))+1, int(torch.max(self.items))+1
        self.test_idx = torch.stack((self.test_users, self.test_items), axis=0)
        self.test_numUsers, self.test_numItems = int(
            torch.max(self.test_users))+1, int(torch.max(self.test_items))+1

        self.rating_interactions = self.vec_to_sparse(
            self.users, self.items, self.ratings,
            size=(max(self.numUsers, self.test_numUsers), max(self.numItems, self.test_numItems)),
            load_full=False,
        )
        self.time_interactions = self.vec_to_sparse(
            self.users, self.items, self.timestamps,
            size=(max(self.numUsers, self.test_numUsers), max(self.numItems, self.test_numItems)),
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

        self.test_rating_interactions = self.vec_to_sparse(
            self.test_users, self.test_items, self.test_ratings,
            size=(max(self.numUsers, self.test_numUsers), max(self.numItems, self.test_numItems)),
            load_full=True,
        )
        self.test_time_interactions = self.vec_to_sparse(
            self.test_users, self.test_items, self.test_timestamps,
            size=(max(self.numUsers, self.test_numUsers), max(self.numItems, self.test_numItems)),
            load_full=False,
        )

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
        solution = self.normalize(self.test_rating_interactions[uID])

        return uID, items, ratings, ratings_togo, time, solution
