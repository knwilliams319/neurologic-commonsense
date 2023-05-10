import json
import pandas as pd
from keyword_extractor import KeywordExtractor

DEV_PATH   = "../data/dev_rand_split.jsonl"
TEST_PATH  = "../data/test_rand_split_no_answers.jsonl"
TRAIN_PATH = "../data/train_rand_split.jsonl"

def process_raw_data( path: str, split: str, extractor_type: str = "keybert", limit:int = 5):
    # create dataframe from json lines
    with open(path) as f:
        lines = f.read().splitlines()

    df = pd.DataFrame(lines)
    df.columns = ['json_element']
    df['json_element'].apply(json.loads)
    df = pd.json_normalize(df['json_element'].apply(json.loads))

    # extract keywords
    ke = KeywordExtractor(extractor_type, limit)
    df["keywords"] = ke.extract_all(list(df["question.stem"]))
    
    # save as csv file
    df.to_csv(f"../data/{split}split.csv")
    
def create_datafiles():
    process_raw_data(DEV_PATH, "DEV")
    process_raw_data(TEST_PATH, "TEST")
    process_raw_data(TRAIN_PATH, "TRAIN")