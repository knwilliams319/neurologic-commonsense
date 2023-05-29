import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

# torch config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models
from models import LanguageModel

# parameters
DROPOUT         = 0.15
MAX_SEQ_LEN     = 64
LR              = 0.0001
BATCH_SIZE      = 128
EPOCHS          = 10
SEED            = 0
MAX_GEN_LEN     = 20
NUM_RETURNS     = 1
NUM_BEAMS       = 5

# MODEL SIZES: 
# { 
#   "gpt2-small" , 
#   "gpt2-medium", 
#   "gpt2-large" 
# }
MODEL_SIZE = "gpt2-small"

# create model
lm = LanguageModel.BaseLM(
    model       = MODEL_SIZE, 
    seed        = SEED,
    max_gen_len = MAX_GEN_LEN,
    num_returns = NUM_RETURNS,
    num_beams   = NUM_BEAMS
)

def train(model: torch.nn.Module, dataset: Dataset, epochs: int, batch_size: int):
  model.train()

  cost = []
  accuracy = []

  for epoch in tqdm(range(epochs)):

    # count = 0
    # correct = 0
    # epoch_loss = 0 

    # for i in range(0, dataset.__len__(), batch_size):
    #   batch = dataset[i]

    #   text = list(batch[:, 0])
    #   labels = batch[:, 1].astype(int)
    #   labels = torch.Tensor(labels).type(torch.LongTensor).to(device)

    #   encoding = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, 
    #                  padding="max_length", return_attention_mask=False, 
    #                  return_tensors="pt")

    #   encoding['input_ids'] = encoding['input_ids'].to(device)

    #   optimizer.zero_grad()

    #   out = model.forward(
    #       encoding['input_ids'],
    #       encoding['input_ids']
    #   )
    #   out = torch.mean(out, dim=1)

    #   probabilities = F.softmax(out, dim=1)
    #   predicted_labels = torch.argmax(probabilities, dim=1)

    #   loss = criterion(out, labels)
    #   epoch_loss += loss.item()
    #   loss.backward()
    #   optimizer.step()

    #   count += batch_size
    #   for i, l in enumerate(labels): 
    #     if predicted_labels[i] == labels[i]:
    #       correct += 1
    
    # cost.append(epoch_loss/count)
    # accuracy.append(correct/count)
    print(f"epoch: {epoch}, accuracy: {accuracy[-1]}, loss: {cost[-1]}")
  
  return cost, accuracy


# SAVE MODEL
TRANSFORMER_SAVE = f"{MODEL_SIZE}.pt"
torch.save(model.state_dict(), TRANSFORMER_SAVE)





# concepts = ["planet", "third", "sun"]
# text = "What is the third planet from the sun?"
# lm = LanguageModel.BaseLM(model="gpt2-medium", max_len=20)
# decoded_text = lm.decode(text=text, concepts=concepts)