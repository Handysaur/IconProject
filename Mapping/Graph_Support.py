
class Node(object):

    def __init__(self, nodeContent) -> None:
        self.content = nodeContent

    def getContent(self):
        return self.content

    def __eq__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        return self.content == __o.getContent()

    def __lt__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        return self.X < __o.getContent()

    def __lt__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        return self.X <= __o.getContent()      

    def __hash__(self) -> int:
        return hash(str(self))

class Arc(object):

    def __init__(self, startNode, destinationNode, cost) -> None:
        self.StartN = startNode
        self.DestN = destinationNode
        self.cost = cost

    def getStartNode(self):
        return self.StartN

    def getDestinationNode(self):
        return self.DestN

    def getCost(self):
        return self.cost                

    def __eq__(self, __o: object) -> bool:
        if type(self) != type(__o):
            return False

        return (self.StartN == __o.getStartNode()) and (self.DestN == __o.getDestinationNode()) and (self.cost == __o.getCost())

    def hasNode(self, node):
        return (node == self.StartN) or (node == self.DestN)  

    def __hash__(self) -> int:
        return hash(str(self))

class Path(object):

    def __init__(self, startNode) -> None:
        self.path = []                         
        self.addNode(startNode)

    def addNode(self, node):
        self.path.append(node)

    def getStartingNode(self):
        return self.path[0]

    def getLastNode(self):
        return self.path[-1]

    def getPath(self):
        return self.path

    def __lt__(self, __o: object) -> bool:
        if (len(self.path) >= len(__o.getPath())):
            return False
        
        for i in range(0, len(self.path)-1):
            if self.path[i] >= __o.getPath()[i]:
                return False

        return True         

    def __le__(self, __o: object) -> bool:
        if (len(self.path) >= len(__o.getPath())):
            return False
        
        for i in range(0, len(self.path)-1):
            if self.path[i] > __o.getPath()[i]:
                return False  

        return True              