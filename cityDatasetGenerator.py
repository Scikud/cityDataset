import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import BallTree
from tqdm import tqdm
import pickle
import random
import heapdict
import sys
sys.path.append("..")


def generateDataset(desiredNumberOfPoints=1e7, connectedRadius=70,
                    source='uscitiesSmall.csv'):
    df = pd.read_csv(source)
    cityGraph = makeCityGraph(df, connectedRadius)
    printSample(cityGraph)
    makeDataset(cityGraph, desiredNumberOfPoints)


def makeCityGraph(df, connectedDistance=70):
    """
    Takes in a dataframe containing the source csv, and a connected Distance 
    (which is the threshold in KM for which we consider two cities to be 
    connected) and generates a graph containing the connected cities
    """

    # Minor preprocessing
    df['lat'], df['lng'] = df['lat'].apply(
        np.radians), df['lng'].apply(np.radians)
    coords = []
    for _, row in df.iterrows():
        lat, lng = float(row['lat']), float(row['lng'])
        coords.append([lat, lng])
    coords = np.array(coords)

    # Make basically a KD tree of city coordinates
    haversineMetric = DistanceMetric.get_metric('haversine')
    coordsBallTree = BallTree(coords, metric=haversineMetric)
    def index2City(x): return ':'.join(
        [str(df.loc[x, :]['city']), str(df.loc[x, :]['state_id'])])

    earthRadius = 6371

    cityGraph = CityGraph()
    print(
        f'Generating City Graph with {connectedDistance} KM connection radius')
    for indx, row in tqdm(df.iterrows()):
        currCityCoords = np.array(
            [float(row['lat']), float(row['lng'])]).reshape(1, 2)

        # Grab list of adjcent city (indices) and their distances away
        adjacentList, distances = coordsBallTree.query_radius(currCityCoords,
                                                              r=connectedDistance/earthRadius, return_distance=True)
        distances = earthRadius * distances

        # Create Vertex for each city, add neighbors
        adjacentList, distances = adjacentList[0],  np.round(
            distances[0], decimals=1)
        neighbors = {int(adjacentList[i]): float(
            distances[i]) for i in range(len(distances))}

        cityVertex = Vertex(id=indx, name=index2City(indx))
        cityVertex.setNeighbors(neighbors)
        cityVertex.setCoordinates(currCityCoords)
        cityGraph.addVertex(indx, cityVertex)

    return cityGraph


def printSample(cityGraph):
    print('Printing Path Sample')
    adjacencyList = cityGraph.grabAdjacenyList()
    cityIndx = np.random.randint(0, len(adjacencyList))
    _, trajectories = dijkstra(cityIndx, adjacencyList)
    pathtraverser = PathTraverser(cityIndx, trajectories)
    paths = pathtraverser.computePaths()
    randomNum = np.random.randint(0, len(paths.items()))
    randomPath = paths[list(paths.keys())[randomNum]]
    humanReadablePath = [cityGraph.getNode(each).name for each in randomPath]
    print(f'Raw Path {randomPath}')
    print(f"Human readable path {'->'.join(humanReadablePath[2:])}")


def makeDataset(cityGraph, numDataPoints):
    """
    Generates the datset, the format is [startCityIndex, EndCityIndex,
    [PathIndices]] where [PathIndicies] is a list of the indexes for cities in 
    the path
    """

    print('Beginning dataset generation')
    adjacencyList = cityGraph.grabAdjacenyList()
    res = []
    seen = set()
    count = 0
    while len(res) < numDataPoints and count < len(adjacencyList):

        # If we're dealing with the full data set, better to pick random cities
        # to make sure we're getting good coverage
        if len(adjacencyList) > 2e5:
            cityIndx = np.random.randint(0, len(adjacencyList))
        else:
            cityIndx = count

        # Compute the paths for the selected city
        _, trajectories = dijkstra(cityIndx, adjacencyList)
        pathtraverser = PathTraverser(cityIndx, trajectories)
        paths = pathtraverser.computePaths()
        for each in paths.keys():
            startCity, endCity = each
            if (startCity, endCity) not in seen and (endCity, startCity) not in seen:
                if len(paths[each]) == 1:
                    paths[each] = [*paths[each]]*2
                res.append([startCity, endCity, *paths[each]])
                res.append([endCity, startCity, *paths[each][::-1]])
                seen.add((startCity, endCity))
        count += 1
        if count % 50 == 0:
            print(f'\rGenerated {len(res)} paths , approximately {round(100* len(res)/numDataPoints,2)}% of desired number of points ...',
                  end='\r')
    random.shuffle(res)
    with open('cityPaths.p', 'wb') as f:
        ix2CityDict, ix2CoordsDict = {}, {}
        for indx in range(len(adjacencyList)):
            ix2CityDict[indx] = cityGraph.getNode(indx).name
            ix2CoordsDict[indx] = cityGraph.getNode(indx).coordinates
        maxCityIndx = len(list(ix2CityDict.keys()))
        pathObject = dict(indexMapping=ix2CityDict, coordsDict=ix2CoordsDict,
                          paths=res, maxCityIndx=maxCityIndx)
        pickle.dump(pathObject, f)

    print(
        f"Generated {len(res)} datapoints of total size {round(sys.getsizeof(res)/(1024**2),2)} MBs ")
    print(f"Saved pickled output at 'cityPaths.p'")


class Vertex():
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.coordinates = None
        self.neighbors = {}

    def setNeighbors(self, neighbors):
        self.neighbors = neighbors

    def setCoordinates(self, coordinates):
        self.coordinates = coordinates


class CityGraph():
    def __init__(self):
        self.vertices = {}

    def addVertex(self, id, vertex):
        self.vertices[id] = vertex

    def getNode(self, id):
        return self.vertices[id]

    def grabAdjacenyList(self,):
        adjencyList = []
        for indx in range(len(self.vertices.items())):
            city = self.vertices[indx]
            adjacent = [[neigbhor, dist]
                        for neigbhor, dist in city.neighbors.items()]
            adjencyList.append(adjacent)
        return adjencyList


def dijkstra(start, edges):
    """ Its ya boy dijkstra. Takes in a start node and computes the distance to all 
    of the nodes in the adjaceny list called 'edges'. Also returns a list called trajectories
    which for each node, stores the 'previous' node to that node following the optimal path
    """

    # Maintain a minheap containing the distaces to different edges, also keep track of trajectories
    minDistancesHeap = heapdict.heapdict(
        {indx: float('inf') for indx in range(len(edges))})
    minDistances = {indx: float('inf') for indx in range(len(edges))}
    trajectory = {indx: None for indx in range(len(edges))}

    # Initialize stuff
    minDistances[start], minDistancesHeap[start] = 0, 0
    trajectory[start] = start

    visited = set()
    while len(minDistances.keys()) > 0:
        currEdge, currDist = minDistancesHeap.popitem()
        if currDist == float('inf'):
            break

        visited.add(currEdge)
        for child, childDist in edges[currEdge]:
            if child in visited:
                continue
            if currDist + childDist < minDistances[child]:
                minDistances[child] = currDist + childDist
                trajectory[child] = currEdge
                minDistancesHeap[child] = currDist + childDist

    return minDistances, trajectory


class PathTraverser():
    """ Class wrapping helper methods which take in the set of previous nodes, here 
        called 'trajectories' and works backwards to compute the full start -> end path    
    """

    def __init__(self, startIndx, trajectories):
        self.start = startIndx
        self.trajectories = trajectories
        self.seen = {startIndx: [startIndx]}

    def backwardsTraverse(self, indx):
        if indx in self.seen:
            return self.seen[indx]
        prev = self.trajectories[indx]
        self.seen[indx] = [indx] + self.backwardsTraverse(prev)
        return self.seen[indx]

    def computePaths(self):
        for indx in reversed(range(len(self.trajectories))):
            if indx not in self.seen and self.trajectories[indx]:
                self.backwardsTraverse(indx)
        start = self.start
        result = {}
        for end, path in self.seen.items():
            path = path[::-1]
            result[(start, end)] = path
        return result


if __name__ == "__main__":
    generateDataset(desiredNumberOfPoints=1e9,
                    connectedRadius=70, source='uscitiesSmall.csv')
