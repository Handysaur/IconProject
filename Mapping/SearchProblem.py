import heapq
from copy import deepcopy

from Mapping import Graph_Support

MAX_PATH_COST = 99999   #Costo fittizzio che indica il già esploramento di un percorso

class SearchProblem(object):

    def __init__(self, nodes, arcs, startNode, goalNode) -> None:
        self.nodes = nodes
        self.arcs = arcs
        self.startNode = startNode
        self.goalNode = goalNode

    def isGoalNode(self, node):
        return node == self.goalNode

    def getNodes(self):
        return self.nodes

    def getArcs(self):
        return self.arcs

    def getStartingNode(self):
        return self.startNode

    def getGoalNode(self):
        return self.goalNode

"""
Frontiera per l'algoritmo A* Star. Implementata con una cosa con priorità poichè restituisce sempre quella con più priorità. In questo caso con distanza minore
"""
class AStar_Frontier(object):

    def __init__(self) -> None:
        self.frontier_Queue = []

    def isEmpty(self):
        return self.frontier_Queue == []

    def addPath(self, pathToAdd, costOfPath):
        heapq.heappush(self.frontier_Queue, (costOfPath, pathToAdd))

    def pop(self):
        return heapq.heappop(self.frontier_Queue)

    def frontierLength(self):
        return len(self.frontier_Queue)

    def iterator(self):
        for (_, p) in self.frontier_Queue:
            yield p

"""
Implementazione dell'algoritmo di ricerca A* Star
"""
def AStar_Alghoritm(resProblem: SearchProblem, heuristic):

    goalNode = resProblem.getGoal()
    startNodes = AStar_Frontier()
    for startNode in resProblem.getStartingNode():
        startNodes.add(Graph_Support.Path(startNode), heuristic(startNode, goalNode))

    node_cameFrom = {}

    #Costo da un certo nodo iniziale fino a quello corrente
    gScore = {}
    for path in startNodes:
        gScore[path.getLastNode()] == 0 #Setto il costo di ciascun nodo a 0

    fScore = {}
    for path in startNodes:
        node = path.getLastNode()
        fScore[node] = heuristic(node, goalNode)

    while len(startNodes) != 0:

        currentNode_Cost, currentNode_Path = startNodes.pop()
        currentNode = currentNode_Path.getLastNode()

        if resProblem.isGoalNode(currentNode):
            return currentNode_Path, currentNode_Cost

        for arc in resProblem.getArcs():
            if (arc.hadNode(currentNode)):
                if arc.getStartNode() == currentNode:   neighbor = arc.getDestinationNode()
                if arc.getDestinationNode() == currentNode:   neighbor = arc.getStartNode()

                currentExploration_gScore = gScore[currentNode] + arc.getCost()
                neighbor_gScore = gScore[neighbor] if (neighbor in gScore) else MAX_PATH_COST    

                #Costo del nodo vicino al nodo corrente è migliore dunque conviene esplorare questo percorso
                if currentExploration_gScore < neighbor_gScore:
                    node_cameFrom[neighbor] = currentNode
                    gScore[neighbor] = currentExploration_gScore

                    fScore[neighbor] = gScore[neighbor] + heuristic(neighbor, goalNode)

                    newPathToExplore = deepcopy(currentNode_Path)
                    newPathToExplore.add(neighbor)

                    if newPathToExplore not in startNodes:
                        startNodes.addPath(newPathToExplore, fScore[neighbor])

    return None                    