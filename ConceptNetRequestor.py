'''
This file contains a class for making requests to the ConceptNet API. There doesn't seem to be a nice Python
package or wrapper for the API, so this class will help us query the API to find edges/relationships between 
concept nodes. 

Reference: https://github.com/commonsense/conceptnet5/wiki/API

Last Modified Date: 5/10/2023
Last Modified By: Kyle Williams
'''
import requests # used by the ConceptNetRequestor
import sys # used by __main__

class ConceptNetRequestor: 
    def __init__(self):
        """
        A class to handle requests to the ConceptNet API.

        Attributes
        ----------
        api : str
            Base path to the concept net API
        """
        self.api = "http://api.conceptnet.io/c/en/"

    def get_edges(self, q_concept: str):
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
        """
        if not q_concept: # Check input for validity
            raise ValueError("cannot query ConceptNet for the empty string")
        
        # Query ConceptNet API and format response into JSON
        # NOTE: Returned edges in the edges list may be in towards the queried concept, or out towards
        #       a related one. Thus, we must check both nodes in the data-processing loop. 
        data = requests.get(self.api + q_concept).json() 

        if 'error' in data.keys(): # This may never happen in practice, but just in case
            raise ValueError(f"queried concept {q_concept} is not a node in ConceptNet") 
        
        edges = [] # Return value
        
        # Data processing loop
        for edge in data['edges']:
            # Even though we're querying the English API path, some concept edges are to foreign languages.
            # These edges should be filtered for the sake of our task.
            if edge['start']['language'] != 'en' or edge['end']['language'] != 'en':
                continue

            edges.append({'start': edge['start']['label'], 
                          'end': edge['end']['label'], 
                          'relationship' : edge['rel']['label'],
                          'weight': edge['weight']}) 
        return edges

if __name__ == "__main__":
    """
    Invoke main from the command line to see the edges associated with an input concept node
    """
    if len(sys.argv) < 2:
        raise ValueError("Please pass an English string to request that node's edges from ConceptNet!")
    elif len(sys.argv) > 2:
        raise ValueError("This function only accepts a single argument")
    else:
        arg = sys.argv[1]
        cnr = ConceptNetRequestor()
        edges = cnr.get_edges(arg)
        for edge in edges:
            print(edge)
        
    
    
