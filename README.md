# Shortest Path Dataset


Shortest Path enumerates the shortest paths between US Cities and is intended to be used as training data for ML work. 


## Generating the dataset

Firt run the cityDatasetGeneratory.py script which should generate a pickled object called cityPaths.p

```bash
python3 cityDatasetGenerator.py
```
There are two controllable parameters here:

* ConnectedRadius is a parameter that controls the radius below which we consider two cities to be connected. For instance a connectedRadius = 70 implies that we consider two cities to be connected if their gps coordinates are within 70 km of each other

* Desired Number of points -- controls the number of shortest path solutions we'd like to generate for the dataset. The main reason this exists is because for the full dataset there are ~30,000 cities and enumerating all possible 900 million paths is computationaly costly, therefore this parameter bounds the number of paths the final output should have.

The pickled object is a dictionary that has 3 keys

* *paths*: An array containing the generated solutions. For each row the first two entries in this array represent the start and end city and the remaining entries represent the solution
* *indexMapping* : A dictionary containing mappings between the indicies used to represent the cities and the corresponding names
* *coordsDict* : A dictionary containing mappings between the city indicies and their latititude and longitude coordinates expressed in radians.


## Unpickling the data

```python
#Quick helper
def getPickleData(path = 'cityPaths.p'):
    with open(path, 'rb') as f:
        return pickle.load(f)

unpickledData = getPickleData(path = 'cityPaths.p')
```


## Example Usage
Here's some code that takes in the unpickled data and constructs a pytorch dataloader

```python
class ShortestPathDataset(Dataset):
    def __init__(self, cityPathData, train=True, maxLength=45):
        #Index --> name and Index-->(lat, lng) coordinates mappinngs
        self.ix2NameDict = cityPathData['indexMapping']
        self.ix2CoordsDict = cityPathData['coordsDict'] #Coords in radians
        
        #Load Paths
        fullPaths = cityPathData['paths']
        self.EoSToken = cityPathData['maxCityIndx']
        self.ix2NameDict[self.EoSToken] = '<EOS>'

        # Fill data array with data
        self.data = [each + [self.EoSToken]
                     for each in fullPaths if len(each[2:]) <= maxLength]

        # 80/20 Training/Test Split
        splitIndx = int(0.8*len(self.data))
        self.dataLen = splitIndx if train else len(self.data) - splitIndx
        self.data = self.data[:splitIndx] if train else self.data[splitIndx:]

    def __len__(self):
        return self.dataLen

    def __getitem__(self, indx):
        return self.data[indx]
```

and some sample usage ...

``` python
#Load in dataset
shortestPathData = ShortestPathDataset(unpickledData)

#Randomly pick the 5th path stored in the data
chosenPath = shortestPathData[4]

#First two entries always represent start and end cities
startCityIndx, endCityIndx = chosenPath[0], chosenPath[1]

#Solution is always stored in the remaining entires
solution = chosenPath[2:]

#Helper for mapping indexes to human readable city names
toName  = lambda x : shortestPathData.ix2NameDict[x]
toCoordinates = lambda x: shortestPathData.ix2CoordsDict[x]

#Print out the names 
print(f'Start City: {toName(startCityIndx)}, End City:{toName(endCityIndx)}')

#print out solution
solutionText = '->'.join([toName(each) for each in solution])
print(f'Solution: {solutionText}')

```

> Start City: Baltimore:MD, End City:Valparaiso:IN

> Solution: Baltimore:MD->Westminster:MD->Shippensburg:PA->Huntingdon:PA->Altoona:PA->Indiana:PA->Butler:PA->Boardman:OH->Tallmadge:OH->Medina:OH->Norwalk:OH->Fremont:OH->Bowling Green:OH->Defiance:OH->Auburn:IN->Warsaw:IN->Plymouth:IN->Valparaiso:IN->\<EOS\>
 

## Some potential helper methods

Below is a quick and dirty collate method (written as a closure for convienence, pytorch however expects just the function) that the dataloader will apply to each batch. Its purpose is two fold:
1. Transform the sequence into effectively a teacher forced language modeling task where for each input sequence we also create target sequence by shifting the input sequence down by one.   
2. Pad each batch with the padding token to ensure that all entries in the batch are the same length   


```python
def makeCollateFn(paddingIndx):
    pIndx = paddingIndx
    def prepareLanguageModelSequence(batch):
        endpoints, paths, lengths = [], [], []
        batchLen = len(batch)
        for each in batch:
            endpoints.append(each[0:2])
            paths.append(each[2:])
            lengths.append(len(each[2:]))
    
        seqLen = max(lengths)
        endpoints = torch.tensor(endpoints).long()
        xxPad  = torch.tensor([pIndx]).new_full((batchLen,seqLen), pIndx)
        yyPad  = torch.tensor([pIndx]).new_full((batchLen,seqLen), pIndx)


        for i, sequence in enumerate(paths):
            xxPad[i, :lengths[i]-1, ...] = torch.tensor(sequence[:-1]).long()
            yyPad[i, :lengths[i]-1, ...] = torch.tensor(sequence[1:]).long()

        return endpoints,xxPad, yyPad
    return prepareLanguageModelSequence
```
