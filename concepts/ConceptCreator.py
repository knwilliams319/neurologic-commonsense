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

DEVPATH = 'DEVsplit.csv'
TESTPATH = 'TESTsplit.csv'
TRAINPATH = 'TRAINsplit.csv'
N_GRAMS = 1

concept_list = {}
CNR = ConceptNetRequestor()

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

for path in [DEVPATH, TESTPATH, TRAINPATH]:
    data_file_path = os.path.join(os.pardir, 'data', path)
    data = pd.read_csv(data_file_path)

    questions_completed = 0

    for keywords_list in data['keywords']:
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



        


