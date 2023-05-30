import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

# torch config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models
from models import LanguageModel
from concepts.ConceptParser import ConceptParser

# paths
DEV = "./data/refinedDEVsplit.csv"
TRAIN = "./data/refinedTRAINsplit.csv"
TEST = "./data/refinedTESTsplit.csv"

# parameters
DROPOUT = 0.15
MAX_SEQ_LEN = 64
LR = 0.0001
BATCH_SIZE = 128
EPOCHS = 1
SEED = 0
MAX_GEN_LEN = 20
NUM_RETURNS = 1
NUM_BEAMS = 5

# number of keywords to retrieve
# concepts for, range: [0-5]
KEYWORD_THRESHOLD = 2

# MODEL SIZES:
# {
#   "gpt2-small" ,
#   "gpt2-medium",
#   "gpt2-large"
# }
MODEL_SIZE = "gpt2-small"
SAVE_PATH = f"trained-{MODEL_SIZE}.pt"

class CommonSenseDataset(Dataset):
  def __init__(self, csv_path: str, batch_size: int):
    super(CommonSenseDataset, self).__init__()
    
    self.data = pd.read_csv(csv_path)
    self._len = self.data.shape[0]
    self.batch_size = batch_size
    self.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    self.has_answer = False
    if "answer" in self.data.columns:
        self.has_answer = True
  
  def __len__(self):
    return self._len
  
  def __getitem__(self, i):
    batch = self.data[i:i+self.batch_size]
    
    questions = list(batch["question.stem"])
    answers   = list(batch["answer"]) if self.has_answer else []
    keywords  = list(batch["keywords"])
    for index, keyword in enumerate(keywords):
        keywords[index] = eval(keyword)

    return questions, answers, keywords

def train(
    model: torch.nn.Module,
    dataset: Dataset,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim,
    criterion: any,
    parser: ConceptParser,
    keyword_threshold: int
):
    """
    Train: Uses cross entropy loss as a training objective. A model is
    considered accurate iff it produces the correct answer token in its 
    text generation sequence.
    """

    model.train()

    cost = []
    accuracy = []

    for epoch in tqdm(range(epochs)):

        count   = 0
        correct = 0
        epoch_loss = 0
        batch_loss = 0
        
        for i in range(0, dataset.__len__(), batch_size):
            
            # shape: (1, BATCH_SIZE)
            questions, answers, keywords = dataset[i]
            
            for j in range(batch_size):
                
                triples = []
                for keyword in keywords[j][:keyword_threshold]:
                    # TODO: get concepts for keyword
                    triples.extend(...)
                    
                # create seed with parsed concepts
                seed = parser.concepts2paragraph(triples)
                
                # add question and answer
                seed += questions[j] + " " + f"({answers[j]})"
                
                # decode
                optimizer.zero_grad()
                pass


            # optimizer.zero_grad()

            # out = model.forward(
            #     encoding['input_ids'],
            #     encoding['input_ids']
            # )
            # out = torch.mean(out, dim=1)

            # probabilities = F.softmax(out, dim=1)
            # predicted_labels = torch.argmax(probabilities, dim=1)

            # loss = criterion(out, labels)
            # epoch_loss += loss.item()
            # loss.backward()
            # optimizer.step()

            # count += batch_size
            # for i, l in enumerate(labels):
            #     if predicted_labels[i] == labels[i]:
            #         correct += 1

        cost.append(epoch_loss/count)
        accuracy.append(correct/count)
        
        print(f"epoch: {epoch}, accuracy: {accuracy[-1]}, loss: {cost[-1]}")

    return cost, accuracy

# TODO: load dataset

# CREATE or LOAD model
model = LanguageModel.BaseLM(
    model=MODEL_SIZE,
    seed=SEED,
    max_gen_len=MAX_GEN_LEN,
    num_returns=NUM_RETURNS,
    num_beams=NUM_BEAMS
)

saved_model = Path(SAVE_PATH)
if saved_model.exists():
    model.load_state_dict(torch.load(SAVE_PATH))
    model.to(device)

# creatw optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# create concept parser
parser = ConceptParser("./utilities.json")

# create dataset
dataset = CommonSenseDataset(DEV, BATCH_SIZE)

# TODO: TRAIN model

# SAVE model
torch.save(model.state_dict(), SAVE_PATH)
