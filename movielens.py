import re
import os
import torch
import numpy as np
import pandas as pd

GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

DEFAULT_HEADER = (
    "userID",
    "itemID",
    "rating",
    "timestamp",
)


class Movielens:
    def __init__(self, sep, item_sep, path, item_path):
        self._sep = sep
        self._item_sep = item_sep
        self._path = path
        self._item_path = item_path

    @property
    def separator(self) -> str:
        return self._sep

    @property
    def path(self) -> str:
        return self._path

    @property
    def item_path(self) -> str:
        return self._item_path

    @property
    def item_separator(self) -> str:
        return self._item_sep


MOVIELENS_100K = Movielens(
    "\t", "|", "ml-100k/u.data", "ml-100k/u.item")


def load_movielens(
    dataset: Movielens = MOVIELENS_100K,
    header: set = DEFAULT_HEADER,
    title_col: str = 'Title',
    genres_col: str = 'Genres',
    year_col: str = 'Year'
) -> pd.DataFrame:

    item_df = load_movielens_items(
        dataset, header, title_col, genres_col, year_col
    )

    df = pd.read_csv(
        dataset.path,
        sep=dataset.separator,
        engine="python",
        names=header,
        usecols=list(range(len(header))),
        header=None,
    )

    # Convert 'rating' type to float
    if len(header) > 2:
        df[header[2]] = df[header[2]].astype(float)

    # Merge rating df w/ item_df
    if item_df is not None:
        df = df.merge(item_df, on=header[1])

    return df


def load_movielens_items(
    dataset: Movielens,
    header: set,
    title_col: str,
    genres_col: str,
    year_col: str
) -> pd.DataFrame:

    movie_col = header[1]
    if not title_col and not genres_col and not year_col:
        return None

    item_header = [movie_col]
    usecols = [0]

    # Year is parsed from title
    if title_col or year_col:
        item_header.append("title_year")
        usecols.append(1)

    genres_header_100k = [str(i) for i in range(19)]
    item_header.extend(genres_header_100k)
    usecols.extend(list(range(5, 24)))  # genres columns

    item_df = pd.read_csv(
        dataset.item_path,
        sep=dataset.item_separator,
        engine="python",
        names=item_header,
        usecols=usecols,
        header=None,
        encoding="ISO-8859-1",
    )

    item_df[genres_col] = item_df[genres_header_100k].values.tolist()
    item_df[genres_col] = item_df[genres_col].map(
        lambda l: "|".join([GENRES[i] for i, v in enumerate(l) if v == 1])
    )

    item_df.drop(genres_header_100k, axis=1, inplace=True)

    if year_col is not None:

        def parse_year(t: str) -> str:
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None

        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)

    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)

    return item_df


def save_to_pickle(df: pd.DataFrame, path: str):
    interactions = df.pivot_table(
        values=['rating', 'timestamp'],
        index=['userID', 'itemID'],
        aggfunc=lambda x: x
    ).sort_values(by=['timestamp'], ascending=True)

    users, items = torch.from_numpy(np.array(interactions.index.codes))
    ratings = torch.from_numpy(interactions['rating'].to_numpy())
    timestamps = torch.from_numpy(interactions['timestamp'].to_numpy())
    torch.save(users, os.path.join(path, 'users.pt'))
    torch.save(items, os.path.join(path, 'items.pt'))
    torch.save(ratings, os.path.join(path, 'ratings.pt'))
    torch.save(timestamps, os.path.join(path, 'timestamps.pt'))


if __name__ == "__main__":
    print("Loading data from Movielens100k...")
    df = load_movielens()
    print("Saving to pickle...")
    save_to_pickle(df, "ml-100k/")
    print("Done!")
