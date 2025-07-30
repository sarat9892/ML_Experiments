import argparse
import json
import os

import pandas as pd


def extract(event):
    df = pd.read_csv(event["source"])
    df.to_csv(os.path.join(event["raw"], "car_sales_raw.csv"), index=False)

    return df

def transform(event, df=None):
    string_cols = event["metadata"]["nominal"]

    df[string_cols] = df[string_cols].apply(lambda x: x.str.lower())
    df.loc[:, 'saledate'] = pd.to_datetime(df['saledate'].str.split('(').str[0].str.strip(),
                                           errors='coerce', utc=True)
    df.to_csv(os.path.join(event["warehouse"], "car_sales_cleaned.csv"), index=False)

    return df

def load(event, df):
    df.dropna(subset=['make', 'model'], how='all', inplace=True)
    df.dropna(subset=['trim', 'model'], how='all', inplace=True)

    df = df.assign(make_model=df["make"] + "-" + df["model"].fillna(''))
    df = df.assign(make_model_trim=df["make_model"] + "-" + df["trim"].fillna(''))
    df.to_csv(os.path.join(event["store"], "car_sales_dataset.csv"), index=False)

    return df

def handler(event):
    car_sales = extract(event)
    clean_car_sales = transform(event, car_sales)
    _ = load(event, clean_car_sales)

    return 0


if __name__ == "__main__":
    default_event = "event/event.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--event", "-e",
                        default=default_event,
                        help="event data file path")
    args = parser.parse_args()

    with open(args.event, "r") as file_handler:
        event_text = file_handler.read()

    event_json = json.loads(str(event_text))

    _ = handler(event_json)
