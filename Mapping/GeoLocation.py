import pandas as pd

from Mapping import Graph_Support

"""
Classe indicante un luogo
"""
class Position(object):

    def __init__(self, x, y, placeName) -> None:
        self.Longitude =  x
        self.Latitude = y
        self.PlaceName = placeName

    def getLongitude(self):
        return self.Longitude

    def getLatitude(self):
        return self.Latitude  

    def getPlaceName(self):
        return self.PlaceName

    def __eq__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        return (self.Longitude == __o.getLongitude()) and (self.Latitude == __o.getLatitude())

    def __lt__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        return (self.Longitude < __o.getLongitude()) and (self.Latitude < __o.getLatitude())

    def __le__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        return (self.Longitude <= __o.getLongitude()) and (self.Latitude < __o.getLatitude())        

    def __hash__(self) -> int:
        return hash(str(self))   

"""
Funzione che dato un file csv contenente le posizioni da caricare, crea un insieme di nodi rappresentanti i vari luoghi e un insieme di archi che rappresentano i collegamenti tra luoghi
"""
def loadPositions(Positions_FilePath):
    nodes = []
    arcs = []
    csv = pd.read_csv(Positions_FilePath, on_bad_lines='skip')

    #Itera su tutte le righe del csv.
    for a, b in csv.iterrows():
        node1 = Graph_Support.Node(Position(b['Lat1'], b['Long1'], b['PlaceName']))
        node2 = Graph_Support.Node(Position(b['Lat2'], b['Long2'], b['PlaceName']))

        if node1 not in nodes: nodes.append(node1)
        if node2 not in nodes: nodes.append(node2)

        arcs.append(Graph_Support.Arc(node1, node2, b['Length']))

    return nodes, arcs    
