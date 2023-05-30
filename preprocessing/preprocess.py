import json
import pandas as pd
import regex as re
from KeywordExtractor import KeywordExtractor

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
    
def apply_json(s):
    try:
        return json.loads(s.replace("\'", "\""))
    except:
        return None

def apply_answer(answer, lst):
    if lst == None:
        return None
    else:
        for item in lst:
            if item["label"] == answer:
                return item["text"]
        return None

def process_data(dir:str, path: str):
    df = pd.read_csv(dir + path)
    
    df["question.choices"] = df["question.choices"].apply(apply_json)
    
    try:
        df["answer"] =  df.apply(lambda r: apply_answer(r["answerKey"], r['question.choices']), axis=1)
        df = df.drop(columns=['answerKey'])
        df = df[df["answer"] != None]
    except:
        pass

    df = df.drop(columns=['Unnamed: 0', 'id', 'question.choices', 'question.question_concept'])
    
    # save as csv file
    df.to_csv(dir + "refined" + path)

def clean_datafiles():
    process_data("../data/", "DEVsplit.csv")
    process_data("../data/", "TESTsplit.csv")
    process_data("../data/", "TRAINsplit.csv")