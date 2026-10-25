
from pymongo import MongoClient
import pandas as pd
from pathlib import Path


def get_db():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['food_waste_db']
    return db


def import_data():
    db = get_db()
    collection = db['food_wastage_data']

    if collection.count_documents({}) == 0:
        csv_path = Path(__file__).resolve().parent.parent / "food_wastage_data.csv"
        df = pd.read_csv(csv_path)

        collection.insert_many(df.to_dict('records'))
        print("Data imported successfully!")
    else:
        print("MongoDB already contains data.")


if __name__ == "__main__":
    import_data()
