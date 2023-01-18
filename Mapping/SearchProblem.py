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

    def __iter__(self):
        for (_, p) in self.frontier_Queue:
            yield p

"""
Implementazione dell'algoritmo di ricerca A* Star
"""
def AStar_Alghoritm(resProblem: SearchProblem, heuristic):

    goalNode = resProblem.getGoalNode()
    startNodes = AStar_Frontier()
    for startNode in resProblem.getStartingNode():
        startNodes.addPath(Graph_Support.Path(startNode), heuristic(startNode, goalNode))

    node_cameFrom = {}

    #gScore rappresenta, per ciascun nodo, il costo dal nodo iniziale fino a quello corrente. Settato inizialmente a 0
    gScore = {}
    for path in startNodes:
        gScore[path.getLastNode()] = 0 #Setto il costo di ciascun nodo a 0
 
    #In fScore si va a rappresentare, per ciascun nodo, il costo effettivo del percorso che va dal nodo di partenza ad un certo nodo n al momento in fase di analisi
    # più una stima, data dall'euristica, che sarà il costo del percorso dal nodo n fino al nodo obiettivo
    fScore = {}
    for path in startNodes:
        node = path.getLastNode()
        fScore[node] = heuristic(node, goalNode)

    #Mentre ci sono ancora nodi nella frontiera
    while startNodes.frontierLength() != 0:

        currentNode_Cost, currentNode_Path = startNodes.pop()   #Percorso dal costo minimo recuperato dalla frontiera
        currentNode = currentNode_Path.getLastNode()

        #Nodo obiettivo trovato, va restituito
        if resProblem.isGoalNode(currentNode):
            return currentNode_Path, currentNode_Cost
        
        for arc in resProblem.getArcs():
            if (arc.hasNode(currentNode)):
                if arc.getStartNode() == currentNode:   neighbor = arc.getDestinationNode()
                if arc.getDestinationNode() == currentNode:   neighbor = arc.getStartNode()

                currentExploration_gScore = gScore[currentNode] + arc.getCost()
                neighbor_gScore = gScore[neighbor] if (neighbor in gScore) else MAX_PATH_COST      #Costo del vicino se non è stato già esplorato

                #Costo del nodo vicino al nodo corrente è migliore dunque conviene esplorare questo percorso
                if currentExploration_gScore < neighbor_gScore:
                    node_cameFrom[neighbor] = currentNode
                    gScore[neighbor] = currentExploration_gScore

                    fScore[neighbor] = gScore[neighbor] + heuristic(neighbor, goalNode)

                    newPathToExplore = deepcopy(currentNode_Path)
                    newPathToExplore.addNode(neighbor)

                    if newPathToExplore not in startNodes:
                        startNodes.addPath(newPathToExplore, fScore[neighbor])

    return None                    