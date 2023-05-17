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


def get_num_relations(edges):
    num_relations = {}
    for edge in edges:
        if edge["relationship"] not in num_relations.keys():
            num_relations[edge["relationship"]] = 1
        else:
            num_relations[edge["relationship"]] += 1
    return num_relations

def process_edges(query_concept, edges_list, num_tripples=5, bidirectional=False):
    # NOTE: Edges will be added bidirectionally. I think it might help to have all outward/inward
    #       nodes attached to a single queried node, and we can still recreate the directional graph
    #       easily if we want.
    if query_concept in concept_list.keys():
        return
    if not bidirectional:
        edges_list = [x for x in edges_list if query_concept == x['start']]

    if num_tripples > len(edges_list):
        num_tripples = len(edges_list)

    scores = [0] * len(edges_list)

    num_relations = get_num_relations(edges_list)
    for index, edge in enumerate(edges_list):
        scores[index] = [edge, edge['weight'] * (len(edges_list)/num_relations[edges_list[index]['relationship']])]
    scores.sort(key=lambda x: x[1])
    concept_list[query_concept] = [i[0] for i in scores[:num_tripples]]


for path in [DEVPATH, TESTPATH, TRAINPATH]:
    data_file_path = os.path.join(os.pardir, 'data', path)
    data = pd.read_csv(data_file_path)

    questions_completed = 0

    for keywords_list in data['keywords']:
        keywords_list = eval(keywords_list)
        if N_GRAMS > 1:  # If we want to try permutations of keywords
            for n_gram in range(2, N_GRAMS + 1):  # Try all permutations of lengths
                for combo in itertools.permutations(keywords_list, n_gram):
                    query_concept = '_'.join(combo)
                    edges = CNR.get_edges(query_concept)
                    if edges: process_edges(' '.join(combo), edges, bidirectional=True, num_tripples=5)

        for keyword in keywords_list:  # Then process the original keywords without permutation
            edges = CNR.get_edges(keyword)
            if edges: process_edges(keyword, edges, bidirectional=True, num_tripples=5)

        questions_completed += 1

        if not questions_completed % 5:
            print(questions_completed)

        with open(f'concepts_{N_GRAMS}_{path.split("split")[0]}', 'w') as f:
            json.dump(concept_list, f)

with open(f'concepts_{N_GRAMS}.json', 'w') as f:
    json.dump(concept_list, f)
