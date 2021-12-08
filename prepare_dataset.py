import numpy as np
from src.datasets.movielens import load_movielens, save_to_pickle
from src.datasets.data_splitters import python_stratified_split

if __name__ == '__main__':
    print("Loading data from Movielens100k...")
    data = load_movielens()

    # Convert the float precision to 32-bit in order to reduce memory consumption
    data['rating'] = data['rating'].astype(np.float32)

    train1, test = python_stratified_split(
        data, ratio=0.75, col_user='userID', col_item='itemID', seed=42
    )
    train2, val = python_stratified_split(
        train1, ratio=0.75, col_user='userID', col_item='itemID', seed=42
    )
    print("Saving to training set pickle...")
    save_to_pickle(train2, "ml-100k/", prefix='train')
    print("Saving to validation set pickle...")
    save_to_pickle(val, "ml-100k/", prefix='val')
    print("Saving to benchmark training set pickle...")
    save_to_pickle(train1, "ml-100k/", prefix='b_train')
    print("Saving to benchmark testing set pickle...")
    save_to_pickle(test, "ml-100k/", prefix='b_test')
    print("Done!")
