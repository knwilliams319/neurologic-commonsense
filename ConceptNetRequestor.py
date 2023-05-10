'''
This file contains a class for making requests to the ConceptNet API. There doesn't seem to be a nice Python
package or wrapper for the API, so this class will help us query the API to find edges/relationships between 
concept nodes. 

Reference: https://github.com/commonsense/conceptnet5/wiki/API

Last Modified Date: 5/10/2023
Last Modified By: Kyle Williams
'''
import requests

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
            The english-language concept word to be queried in ConceptNet

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
        
        data = requests.get(self.api + q_concept).json() # Query ConceptNet API and format response into JSON

        if 'error' in data.keys(): # This may never happen in practice, but just in case
            raise ValueError(f"queried concept {q_concept} is not a node in ConceptNet") 
        
        if len(data['edges']) == 0: # Return empty list if there are no edges
            return []
        
        # TODO: This code does nothing to remove repeat edges. Should we just keep the repeat edge with the highest
        #       weight? Should we average the weights? Or should we keep the edge that's the most recently updated?
        #
        #       We should investigate to the significance of weights before we make a decision.
        edges = [{'start': q_concept, 
                  'end': edge['end']['label'], 
                  'relationship' : edge['rel']['label'],
                  'weight': edge['weight']} for edge in data['edges']] # Return value
        return edges

if __name__ == "__main__":
    cnr = ConceptNetRequestor()

    # This is an interesting case of ConceptNet. If you look at the output, there are numerous repeat edges
    # (smile, HasSubevent, smile) with different weights. 
    print(cnr.get_edges("smile"))
