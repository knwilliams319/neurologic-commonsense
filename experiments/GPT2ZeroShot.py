'''
This file contains a scrpt for finding the zero-shot performance of GPT2 on CommonsenseQA. 
This script produces two outputs:

Accuracy: if GPT-2's generated answer contains the text of the correct option, then we count it as correct
? : i want to get accuracy working before I try to make any other metrics

Last Modified Date: 5/21/2023
Last Modified By: Kyle Williams
'''

# Necessary imports
import sys
from LanguageModel import BaseLM
import pandas as pd
import numpy as np
import os
import json

# Main script
if __name__ == "__main__":
    # Load GPT2 with a set seed
    '''
    TODO: We should investigate how much the decoding hyperparameters make a difference.
          Stuff like 'max_len', 'num_beams', etc. for beam search. We should also report
          the scores for greedy decoding and other methods. 
    '''
    gpt2 = BaseLM('gpt2')

    # Load the dev split (can't report data for test set without answers)
    cwd = os.getcwd()
    parent_path = '/'.join(cwd.split('/')[0:-1]) # removes the innermost folder (currently /experiments)
    dev = pd.read_csv(parent_path + '/data/DEVsplit.csv')
    dev = dev.drop(columns = ['Unnamed: 0']) # the CSVs were saved with a leading index column that we can ignore

    # Keep track of model's answers
    answers = ['' for _ in range(dev.shape[0])]

    # Query the model for each of its answers
    for i, row in dev.iterrows():
        '''
        Create the prompt for the model. They will look like the following example (without a newline):

        A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? 
        A: bank. B: library. C: department store. D: mall. E: new york.
        '''
        prompt = row['question.stem']
        
        # Load the row. they were saved as strings, so this is a little wonky. I decided to use
        # json.loads, which expects double quoted property keys. Since the question stem was saved
        # as one huge json string with single quoted keys, we have to be careful to overwrite these 
        # without blindly overwriting single quotes in the choices (e.g. inside a contraction)
        choices_str = row['question.choices']
        choices_str = choices_str.replace("'label'", '"label"')
        choices_str = choices_str.replace("'text'", '"text"')
        choices_str = choices_str.replace('"label": \'', '"label": "')
        choices_str = choices_str.replace('"text": \'', '"text": "')
        choices_str = choices_str.replace('\', "text"', '", "text"')
        choices_str = choices_str.replace('\'}', '"}')
        choices = json.loads(choices_str)

        for choice in choices: # Append the choices to the prompt
            prompt += f" {choice['label']}: {choice['text']}."
        
        # print(prompt)
        
