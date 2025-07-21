import argparse
import pandas as pd


def extract(input_files):
    red = pd.read_csv(input_files["red_wine"], sep=";")
    white = pd.read_csv(input_files["white_wine"], sep=";")

    red.to_csv("data/raw/red.csv", index=False)
    white.to_csv("data/raw/white.csv", index=False)

    return red, white

def transform(red_df, white_df):
    red_df.loc[:, "red"] = 1
    red_df.loc[:, "white"] = 0

    white_df.loc[:, "red"] = 0
    white_df.loc[:, "white"] = 1

    wine = pd.concat([red_df, white_df], axis=0, ignore_index=True)

    return red_df, white_df, wine

def load(red_df, white_df, wine):

    red_df.to_csv("data/processed/red.csv", index=False)
    white_df.to_csv("data/processed/white.csv", index=False)
    wine.to_csv("data/processed/wine.csv", index=False)

    return "data/processed/wine.csv"

def handler(data_files):
    red_wine, white_wine = extract(data_files)
    red_t, white_t, wine_t = transform(red_wine, white_wine)
    processed_file = load(red_t, white_t, wine_t)

    return processed_file


if __name__ == "__main__":
    input_paths = {"red_wine": "data/input/winequality-red.csv",
                   "white_wine": "data/input/winequality-white.csv"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i",
                        default=input_paths,
                        help="Dictionary of data file paths")
    args = parser.parse_args()

    _ = handler(args.inputs)
