from src.datasets.dataloader import UserDataset

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
    uID, items, ratings, ratings_togo, time, solution = user_dataset[0]

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
            uID, items, ratings, ratings_togo, time, solutions = batch
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
            print(
                f"user_solutions: {solutions.size()}, # solutions: {(solutions>=0).sum()}")
            print(solutions)
            print()
        else:
            break
