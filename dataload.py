import pandas
from typing import List
import pathlib


def default_path():

    path = pathlib.Path(__file__)
    path = path.parent / "data/weather_data.csv"

    return path.__str__()


def weather_dataframe(csv_path: "str" = default_path()) -> "pandas.DataFrame":

    # Load dataset as pandas DataFrame
    dataset: "pandas.DataFrame" = pandas.read_csv(csv_path)

    # Set list of redundant columns
    empty_column_names: "List[str]" = [
        "Unnamed: 4",
        "Unnamed: 8",
        "Unnamed: 10",
        "Unnamed: 12",
        "Unnamed: 14",
        "Unnamed: 16",
        "Unnamed: 18",
        "Unnamed: 20",
        "Unnamed: 22",
        "Unnamed: 24",
        "Unnamed: 26",
        "Unnamed: 28",
        "Unnamed: 30",
        "Unnamed: 32",
    ]

    # Drop redundant columns
    dataset = dataset.drop(labels=empty_column_names, axis=1)

    # Set columns to names given in first row
    dataset.columns = dataset.iloc[0].to_list()

    # Remove first row
    dataset = dataset.iloc[1:]

    return dataset


if __name__ == "__main__":

    print(default_path())
