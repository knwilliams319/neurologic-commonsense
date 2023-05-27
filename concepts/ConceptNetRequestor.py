'''
This file contains a class for making requests to the ConceptNet API. There doesn't seem to be a nice Python
package or wrapper for the API, so this class will help us query the API to find edges/relationships between 
concept nodes. 

Reference: https://github.com/commonsense/conceptnet5/wiki/API

Last Modified Date: 5/10/2023
Last Modified By: Kyle Williams
'''
# Query methods
import requests 
import re 
import dask.dataframe as dd 
import pandas as pd
import time

import sys # used by __main__ only


class ConceptNetRequestor: 
    def __init__(self, mode='csv'):
        """
        A class to handle requests to the ConceptNet API.

        Attributes
        ----------
        mode : str 
             Defaults to 'csv' to use dask and pandas to read from the cleaned CSV of ConceptNet
             edges. Otherwise, will query the API through an HTTP request
        api : str
            Base path to the concept net API
        """
        self.mode = mode
        if mode == 'csv':
            self.edges_out = dd.read_csv('conceptnet-out-assertions-5.7.0-en.csv', sep=',').set_index('src')
            self.edges_in = dd.read_csv('conceptnet-in-assertions-5.7.0-en.csv', sep=',', dtype={'dst': 'object'}).set_index('dst')
        else:
            self.api = "http://api.conceptnet.io/c/en/"

    def get_edges(self, q_concept: str, direction:str='both', raise_error=False):
        """
        Makes a request for the information contained at a ConceptNet node, then returns a list of the
        edges of that node.

        Parameters
        ----------
        q_concept: str
            The english-language concept word to be queried in ConceptNet. Multi-word concepts can be queried
            with underscores between words, e.g. "apartment_building"

        Returns
        -------
        edges : List[Dict]
            A list of concepts connected to the input 'q_concept' by edges in ConceptNet. This list may be empty 
            if the queried concept has no edges. Each returned concept is represented by dictionary with the
            following structure:
            {
                'start' --> str: the queried concept that resulted in this concept being found
                'end' --> str: the new concept connected to the queried concept by an edge
                'relationship' --> str: the relationship that connects the queried concept and this one
                'weight' --> float: the weight of the edge
            }
        raise_error : Bool
            Whether or not this function should raise a ValueError if the queried concept is not a node in
            ConceptNet
        """
        if self.mode == 'csv': return self._get_edges_csv(q_concept, direction, raise_error=raise_error)
        else: return self._get_edges_api(q_concept, raise_error=raise_error)

    def _get_edges_csv(self, q_concept:str, direction:str, raise_error:bool=False):
        # TODO: Finish writing this function, using dask and the src index as to query ConceptNet
        #       in a way that is hopefully very efficient. To get both directions efficiently, I
        #       should consider using a second version that is indexed on dst
        if direction == 'both':
            t1 = time.time()
            nodes_out = self.edges_out.loc[q_concept].compute()['dst'].tolist()
            nodes_in = self.edges_in.loc[q_concept].compute()['src'].tolist()
            print (time.time() - t1)
            return nodes_out + nodes_in
    
    def _get_edges_api(self, q_concept: str, raise_error=False):
        if not q_concept: # Check input for validity
            raise ValueError("cannot query ConceptNet for the empty string")
        
        # Query ConceptNet API and format response into JSON
        # NOTE: Returned edges in the edges list may be in towards the queried concept, or out towards
        #       a related one. Thus, we must check both nodes in the data-processing loop. 
        data = requests.get(self.api + q_concept).json() 

        if raise_error and 'error' in data.keys(): # This may never happen in practice, but just in case
            raise ValueError(f"queried concept {q_concept} is not a node in ConceptNet")
        
        edges = [] # Return value
        
        # Data processing loop
        for edge in data['edges']:
            # Even though we're querying the English API path, some concept edges are to foreign languages.
            # These edges should be filtered for the sake of our task.

            if edge['surfaceText'] == None: continue 
            
            if edge['start']['language'] != 'en' or edge['end']['language'] != 'en':
                continue

            edges.append({'start': re.sub(r"-", " ", edge['start']['label']), 
                          'end': re.sub(r"-", " ", edge['end']['label']), 
                          'relationship' : edge['rel']['label'],
                          'weight': edge['weight']}) 
        return edges

if __name__ == "__main__":
    """
    Invoke main from the command line to see the edges associated with an input concept node
    """
    if len(sys.argv) < 4:
        raise ValueError("Please pass an English string to request that node's edges from ConceptNet!")
    elif len(sys.argv) > 4:
        raise ValueError("This function only accepts a single argument")
    else:
        mode = sys.argv[1]
        arg1 = sys.argv[2]
        arg2 = sys.argv[3]
        cnr = ConceptNetRequestor(mode)
        edges = cnr.get_edges(arg1, arg2)
        for edge in edges:
            print(edge)
        
    
    
