import pickle
import base64
import pandas as pd


def encode_dataframe(df: pd.DataFrame) -> str:
    pickled = pickle.dumps(df)
    pickled_b64 = base64.b64encode(pickled)
    hug_pickled_str = pickled_b64.decode("utf-8")
    return hug_pickled_str


def decode_dataframe(data: str) -> pd.DataFrame:
    return pickle.loads(base64.b64decode(data.encode()))
