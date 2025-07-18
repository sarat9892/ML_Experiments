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


if __name__ == "__main__":

    input_paths = {"red_wine": "D:/Projects/Datasets/data/wine_quality/winequality-red.csv",
                   "white_wine": "D:/Projects/Datasets/data/wine_quality/winequality-white.csv"}

    red_wine, white_wine = extract(input_paths)
    red_t, white_t, wine_t = transform(red_wine, white_wine)
    load(red_t, white_t, wine_t)
