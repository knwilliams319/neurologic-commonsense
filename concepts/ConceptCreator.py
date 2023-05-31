'''
This script creates a JSON dictionary mapping input node strings to output nodes 
found in the edges from a ConceptNet query. It will contain the related concept strings for all
three sets, giving us the effective knowledge graph we need for the CommonsenseQA dataset. 
'''
import os
import pandas as pd
import itertools
from ConceptNetRequestor import ConceptNetRequestor
import json
import pickle

# Constants 
DEVPATH = 'DEVsplit.csv'      # Name of dev set CSV file with keywords extracted
TESTPATH = 'TESTsplit.csv'    # Name of test set CSV file with keywords extracted
TRAINPATH = 'TRAINsplit.csv'  # Name of train set CSV file with keywords extracted
DEPTH = 1                     # Depth of ConceptNet edge traversal 
N_GRAMS = 1                   # Query ConceptNet with combinations of keywords of this size
CNR = ConceptNetRequestor()   # Our interface for querying data from the HTTP ConceptNet API

# TODO: Move initialization of this into a main function
concept_list = {}

def find_vocab_size():
    '''
    Returns the vocabulary size of the effective knowledge graph created from the keyword concepts extracted from
    the questions of CommonsenseQA. An adjacency matrix is probably the cheapest way to store our effective knowledge
    graph, but to create one, we must know the vocabulary size beforehand. 
    '''
    vocab = {}  # Maps keywords to their 0-indexed position in the adjacency matrix
    depths = {} # Tracks at what depth a keyword was added so we can speed up graph search

    for path in [DEVPATH]: #, TESTPATH, TRAINPATH]: # Loop over all CommonsenseQA files
        cwd = os.getcwd()
        data = pd.read_csv(cwd + f'/data/{path}')
        data = data.drop(columns = ['Unnamed: 0']) # the CSVs were saved with a leading index column that we can ignore

        questions_completed = 1

        for keywords_list in data['keywords']: # Each question has its own list of keywords extracted by BERT
            keywords_list = eval(keywords_list)

        if N_GRAMS > 1: # If we want to try permutations of keywords
            for n_gram in range(2, N_GRAMS+1): # Try all lengths of permutations specified
                for combo in itertools.permutations(keywords_list, n_gram):
                    query_concept = '_'.join(combo) # Multi-word concepts are separated by '_' in the API path
                    _add_to_vocab(vocab, query_concept, depths, DEPTH)

        for keyword in keywords_list: # Then process the original keywords without permutation
            _add_to_vocab(vocab, keyword, depths, DEPTH)

        print(questions_completed)
        questions_completed += 1

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("depths.pkl", "wb") as f:
        pickle.dump(depths, f)

    print(len(vocab.keys()))
    return len(vocab.keys())


def _add_to_vocab(vocab, keyword, depths, curr_depth):
    '''
    Helper function to add 'keyword' to our vocabulary. Optionally explores edges of 'keyword' from ConceptNet
    if 'depth' is nonzero. When this function adds 'keyword' to the vocabulary, it takes note of the index
    of the row/column this keyword will be mapped to in the adjacency matrix to be created after. 

    NOTE: This function maintains the invariant that any item in 'vocab' is also in 'depths'
    '''
    # BASE CASES: Return early if these are hit
    # CASE 1) If 'curr_depth' is 0, we are simply adding this node without traversing its edges
    # CASE 2) If we've seen this keyword, check 'depths' to see if we've already done the work

    # CASE ELSE) This keyword is not-yet-seen, we need to add it and find its edges because 'curr_depth' is nonzero

    if curr_depth == 0: # CASE 1
        if keyword not in vocab:
            idx = len(vocab) # this ultimately maps keywords to rows/columns of the adjacency matrix
            vocab[keyword] = idx
            depths[keyword] = curr_depth
        return
    elif keyword in depths: # CASE 2
        if depths[keyword] >= curr_depth: return
        else: depths[keyword] = curr_depth # 'curr_depth' is greater than tracked depth, so we must update and do work
    else: # CASE ELSE
        idx = len(vocab)
        vocab[keyword] = idx
        depths[keyword] = curr_depth
    
    # RECURSIVE STEP: query ConceptNet for edges in/out of this node. Then add it and its connected concepts.
    edges = CNR.get_edges(keyword) # Might be empty if this keyword is not an actual node in ConceptNet
    if edges:
        for edge in edges:               # Then add the connected concepts recursively
            if keyword == edge['start']: _add_to_vocab(vocab, edge['end'], depths, curr_depth-1)
            else: _add_to_vocab(vocab, edge['start'], depths, curr_depth-1)
    else: # since this is not an actual node in ConceptNet, we should not track it (may have been added by CASE ELSE)
        del vocab[keyword]
        del depths[keyword]

def process_edges(query_concept, edges_list):
    # NOTE: Edges will be added bidirectionally. I think it might help to have all outward/inward
    #       nodes attached to a single queried node, and we can still recreate the directional graph
    #       easily if we want. 

    for edge in edges_list:
        if edge['start'] not in concept_list:
            # Each item in concept_list gives an array of concept edges pointing in/out 
            # from the key concept. The corresponding edges' relations and weights will be
            # stored in the same order in case they end up being useful later. 
            concept_list[edge['start']] = {
                'concepts_in': [],
                'rlns_in': [],
                'weights_in': [],
                'concepts_out': [],
                'rlns_out': [],
                'weights_out': []
            }

        if edge['end'] not in concept_list:
            # Each item in concept_list gives an array of concept edges pointing in/out 
            # from the key concept. The corresponding edges' relations and weights will be
            # stored in the same order in case they end up being useful later. 
            concept_list[edge['end']] = {
                'concepts_in': [],
                'rlns_in': [],
                'weights_in': [],
                'concepts_out': [],
                'rlns_out': [],
                'weights_out': []
            }

        if query_concept == edge['start']: # If the key is the start node, this is an outward edge
           
            # Make sure the edge isn't already tracked, perhaps by another question that had
            # the same keyword.
            if edge['end'] not in concept_list[edge['start']]['concepts_out']: 
                concept_list[edge['start']]['concepts_out'].append(edge['end'])
                concept_list[edge['start']]['rlns_out'].append(edge['relationship'])
                concept_list[edge['start']]['weights_out'].append(edge['weight'])

            if edge['start'] not in concept_list[edge['end']]['concepts_in']:
                concept_list[edge['end']]['concepts_in'].append(edge['start'])
                concept_list[edge['end']]['rlns_in'].append(edge['relationship'])
                concept_list[edge['end']]['weights_in'].append(edge['weight'])

        else:  # This is an inward edge

            # Make sure the edge isn't already tracked, perhaps by another question that had
            # the same keyword.
            if edge['start'] not in concept_list[edge['end']]['concepts_in']:
                concept_list[edge['end']]['concepts_in'].append(edge['start'])
                concept_list[edge['end']]['rlns_in'].append(edge['relationship'])
                concept_list[edge['end']]['weights_in'].append(edge['weight'])

            if edge['end'] not in concept_list[edge['start']]['concepts_out']:
                concept_list[edge['start']]['concepts_out'].append(edge['end'])
                concept_list[edge['start']]['rlns_out'].append(edge['relationship'])
                concept_list[edge['start']]['weights_out'].append(edge['weight'])

'''
for path in [DEVPATH, TESTPATH, TRAINPATH]:
    data_file_path = os.path.join(os.pardir, 'data', path)
    data = pd.read_csv(data_file_path)

    questions_completed = 0

    for keywords_list in data['keywords']:
        keywords_list = eval(keywords_list)
        if N_GRAMS > 1: # If we want to try permutations of keywords
            for n_gram in range(2, N_GRAMS+1): # Try all permutations of lengths
                for combo in itertools.permutations(keywords_list, n_gram):
                    query_concept = '_'.join(combo)
                    edges = CNR.get_edges(query_concept)
                    if edges: process_edges(' '.join(combo), edges)

        for keyword in keywords_list: # Then process the original keywords without permutation
            edges = CNR.get_edges(keyword)
            if edges: process_edges(keyword, edges)

        questions_completed+=1

        if questions_completed % 100:
            print(questions_completed)

            with open(f'concepts_{N_GRAMS}.json', 'w') as f:
                json.dump(concept_list, f)

with open(f'concepts_{N_GRAMS}.json', 'w') as f:
    json.dump(concept_list, f)
'''

if __name__ == "__main__":
    find_vocab_size()





