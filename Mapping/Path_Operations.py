import math
import folium
import webbrowser

from Mapping import SearchProblem, GeoLocation

import os
SYSTEM_ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('IconProject'))))

# File da cui caricare la mappa
MAP_PATH = SYSTEM_ROOTDIR + "\\Documents\\IconProject\\Mapping\\Altamura.csv"

"""
Creazione di una mappa su file system, dato un file csv, passato in input
"""
def createMap(mapPath):
    points = []
    streets = []

    points, streets = GeoLocation.loadPositions(mapPath)

    map = folium.Map(location=[points[0].getContent().getLatitude(), points[0].getContent().getLongitude()], zoom_start=12)

    for point in points:
        if 'Medico' in point.getContent().getPlaceName():
            folium.Marker(location = [point.getContent().getLatitude(), point.getContent().getLongitude()], tooltip = point.getContent().getPlaceName(), icon = folium.Icon(color = 'blue', icon = "fas fa-user-doctor", prefix = "fa")).add_to(map)
        if 'Ospedale' in point.getContent().getPlaceName():
            folium.Marker(location = [point.getContent().getLatitude(), point.getContent().getLongitude()], tooltip = point.getContent().getPlaceName(), icon = folium.Icon(color = 'blue', icon = "fas fa-hospital", prefix = "fa")).add_to(map)    
        folium.Marker(location = [point.getContent().getLatitude(), point.getContent().getLongitude()], tooltip = point.getContent().getPlaceName()).add_to(map)

    for street in streets:
        folium.PolyLine(locations = [(street.getStartNode().getContent().getLatitude(), street.getStartNode().getContent().getLongitude()), (street.getDestinationNode().getContent().getLatitude(), street.getDestinationNode().getContent().getLongitude())], color = 'orange').add_to(map)    

    map.save("lastMap.html") 

def showMap(mapPath):
    webbrowser.open_new_tab(mapPath)

"""
Funzione che calcola il percorso tra due posizioni, andando a sfruttare l'algoritmo A* per risolvere un problema di ricerca
Viene fornito in input il percorso ad un file csv che contiene i dati dei percorsi
La funzione poi va a creare un problema di ricerca e lo risolve attraverso l'uso dell'algoritmo A*
Infine la mappa viene salvata su file system e viene restituito il costo del percorso trovato
"""
def findPath(startPosition: GeoLocation.Position, goalPosition: GeoLocation.Position, mapPath):
    nodes = []
    arcs = []
    nodeStart = []
    goalNode = None

    nodes, arcs = GeoLocation.loadPositions(mapPath)

    for node in nodes:
        if node.getContent().getLongitude() == startPosition.getLongitude() and node.getContent().getLatitude() == startPosition.getLatitude() and not nodeStart:
            nodeStart.append(node)

        if node.getContent().getLongitude() == goalPosition.getLongitude() and node.getContent().getLatitude() == goalPosition.getLatitude() and goalNode == None:
            goalNode = node

    searchProblem = SearchProblem.SearchProblem(nodes, arcs, nodeStart, goalNode)

    path, pathCost = SearchProblem.AStar_Alghoritm(searchProblem, eucledian_Heuristic)

    map = folium.Map(location=[nodeStart[0].getContent().getLatitude(), nodeStart[0].getContent().getLongitude()], zoom_start=12)

    positionNodes = path.getPath()

    #Impostazione di un marker personalizzato per la posizione di destinazione, sia essa un medico o un ospedale
    if 'Medico' in positionNodes[-1].getContent().getPlaceName():
        folium.Marker(location = [positionNodes[-1].getContent().getLatitude(), positionNodes[-1].getContent().getLongitude()], tooltip = positionNodes[-1].getContent().getPlaceName(), icon = folium.Icon(color = 'green', icon = "fas fa-user-doctor", prefix = "fa")).add_to(map)
    if 'Ospedale' in positionNodes[-1].getContent().getPlaceName():
        folium.Marker(location = [positionNodes[-1].getContent().getLatitude(), positionNodes[-1].getContent().getLongitude()], tooltip = positionNodes[-1].getContent().getPlaceName(), icon = folium.Icon(color = 'green', icon = "fas fa-hospital", prefix = "fa")).add_to(map)   

    #Impostazione di un marker personalizzato per la posizione di partenza
    folium.Marker(location = [positionNodes[0].getContent().getLatitude(), positionNodes[0].getContent().getLongitude()], tooltip = positionNodes[0].getContent().getPlaceName(), icon = folium.Icon(color = 'yellow', icon = "fas fa-person-walking", prefix = "fa")).add_to(map)

    #Aggiunta dei singoli nodi nel percorso
    for node in positionNodes[1:-1]:
        folium.Marker(location = [node.getContent().getLongitude(), node.getContent().getLongitude()], tooltip = node.getContent().getPlaceName()).add_to(map)             

    #Ciclo su tutti i nodi per creare la strada, andando ad evidenziarla
    for i in range(0, len(positionNodes) - 1):
        folium.PolyLine(locations = [(positionNodes[i].getContent().getLatitude(), positionNodes[i].getContent().getLongitude()), (positionNodes[i + 1].getContent().getLatitude(), positionNodes[i + 1].getContent().getLongitude())], color = 'orange').add_to(map)        

    map.save("lastMap.html")
    return pathCost

"""
Implementazione di uan funzione euristica utilizzata all'interno delle ricerche euristiche
L'implementazione adottata è quella della distanza euclidea che risulta ottimale per il calcolo della distanza in linea d'aria tra due coordinate
Funzione ammissibile e dunque non sovrastima il costo effettivo della distanza
"""
def eucledian_Heuristic(positionA: GeoLocation.Position, positionB: GeoLocation.Position):
    return math.sqrt((positionA.getContent().getLatitude() - positionB.getContent().getLongitude()) **2 + abs(positionA.getContent().getLatitude() - positionB.getContent().getLatitude())**2)

createMap(MAP_PATH)

showMap("lastMap.html")    