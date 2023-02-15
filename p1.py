
import math
import copy
import queue
import random
from queue import PriorityQueue

class boardObject:
    def __init__(self, dataval):
        self.dataval = dataval# 2d array that present the board
        self.gScore = None
        self.cameFrom = None # the object of the "father"
        self.fScore = None
        self.hScore = None
        self.my3TopNeighbor = None
        self.probability = None
        self.vScoreForGenetic = None #for genetic
        self.pScoreForGenetic = None#for genetic
        self.ifMutate = False#for genetic
        self.myMom = None#for genetic
        self.myDad = None#for genetic

    def __lt__(self, _):  # implement of operator
        return True

def find_path(startingBoard, goalBoard, search_method, detail_output):
    if search_method == 1 and not detail_output:
        printWithNoHeuristic(startingBoard, goalBoard, 1)
    elif search_method == 1 and detail_output:
        printWithHeuristic(startingBoard, goalBoard)
    elif search_method == 2:
        printWithNoHeuristic(startingBoard, goalBoard, 2)
    elif search_method == 3 and not detail_output:
        printWithNoHeuristic(startingBoard, goalBoard, 3)
    elif search_method == 3 and detail_output:
        printWithProbability(startingBoard, goalBoard)
    elif search_method == 4 and not detail_output:
        printWithNoHeuristic(startingBoard, goalBoard, 4)
    elif search_method == 4 and detail_output:
        printWithBag(startingBoard, goalBoard)
    elif search_method == 5 and not detail_output:
        printWithNoHeuristic(startingBoard, goalBoard, 5)
    elif search_method == 5 and detail_output:
        printWithGenericCreation(startingBoard, goalBoard)

def printWithGenericCreation(startingBoard, goalBoard):
    sb = boardObject(startingBoard)
    path = myGenetic(sb, goalBoard)
    if len(path) == 1:
        print("No path found.")
    else:
        counter = 1
        for board in path:
            if board.dataval == startingBoard:
                print("Board 1 (starting position):")
                print2dArray(board.dataval)
            elif board.dataval == goalBoard:
                print("Board " + str(counter) + " (goal position):")
                print2dArray(board.dataval)
            else:
                print("Board " + str(counter) + ":")
                print2dArray(board.dataval)
                print("-----")
            counter += 1
    if len(path) > 1:
        print("Starting board 1 (probability of selection from population:: " + str(path[1].myDad.pScoreForGenetic) + "):")
        print2dArray(path[1].myDad.dataval)
        print("-----")
        print("Starting board 2 (probability of selection from population:: " + str(path[1].myMom.pScoreForGenetic) + "):")
        print2dArray(path[1].myMom.dataval)
        print("-----")
        if path[1].ifMutate:
            print("Result board (mutation happened::yes):")
        else:
            print("Result board (mutation happened::no):")
        print2dArray(path[1].dataval)
    else:
        print("Starting board 1 (probability of selection from population:: " + str(path[0].myDad.pScoreForGenetic) + "):")
        print2dArray(path[0].myDad.dataval)
        print("-----")
        print("Starting board 2 (probability of selection from population:: " + str(path[0].myMom.pScoreForGenetic) + "):")
        print2dArray(path[0].myMom.dataval)
        print("-----")
        if path[0].ifMutate:
            print("Result board (mutation happened::yes):")
        else:
            print("Result board (mutation happened::no):")
        print2dArray(path[0].dataval)

def myGenetic(startingBoard, goalBoard):
    firstPopulation = findAllOptions(startingBoard.dataval)
    firstPopulationPriorityQ = queue.PriorityQueue()
    childForPrint = None
    for board in firstPopulation: #set value in the firstPopulation
        board.hScore = findHeuristic(board.dataval, goalBoard)#score for every option
        board.cameFrom = startingBoard
        firstPopulationPriorityQ.put((board.hScore, board))
        if board.dataval == goalBoard: #check if one of then is the goal
            reconstruct_path(board)
    currentPopulation = []
    if len(firstPopulation) > 10:#choose only the best 10 neighbors if the firstPopulation gigger than 10
        for i in range(10):
            currentPopulation.append(firstPopulationPriorityQ.get()[1])#get the 10 good parents
            if currentPopulation[i].hScore == math.inf:
                currentPopulation.remove(currentPopulation[i])
                break
    else:
        currentPopulation = firstPopulation
    counter = 0
    while counter < 500: #limit to 500 generation
        if counter != 0:#set the vaues for the children
            for board in currentPopulation:
                board.hScore = findHeuristic(board.dataval, goalBoard)  # score for every option, no need for the firt iteration
        giveProbability(currentPopulation)# set v and p value (as we learned in tapi course
        nextGeneration = []
        for i in range(len(currentPopulation)):#create 10 childres ro less if in the fort population I have less that 10 neighbors
            giveProbability(currentPopulation)
            myRanges = giveRange(currentPopulation)# returnt array of element that every elemen contain the boardobj,LB, UB
            newChild = None
            cnt = 0
            pairs = []
            while newChild == None and cnt < 200:#try to create until 200 pair of parents
                parents = setMyParents(myRanges)#return array of 2 element, mom & dad
                if parents not in pairs and len(parents)>0:# to avoid double checks
                    pairs.append(parents)
                    newChild = createChild(parents)#return valid child or null
                    if childForPrint is None:
                        childForPrint = newChild
                cnt = cnt + 1
            if newChild != None:
                if newChild.dataval == goalBoard:
                    return reconstruct_path(newChild)
                if newChild.hScore != math.inf:
                    nextGeneration.append(newChild)
        currentPopulation = nextGeneration#switch the generation
        counter = counter + 1
    return [childForPrint] #if no path found return emapy array

def createChild(parents):
    counter = 0
    child = None #initialze to null
    while counter < 40: #why 40? there is 6 option to cut so i gave upperbound very friendky, and if after 100 attemps probably there is no valid child from those parents
        cutOfBoard = random.randint(1,6)#choose the location of the cutting of the board
        dataval = []
        for i in range(cutOfBoard):
            dataval.append(parents[0].dataval[i])
        for j in range(6-cutOfBoard):
            dataval.append(parents[1].dataval[j+cutOfBoard])
        ran = random.random() #random element for create a mutate, my mutate is choose random nighber of one of the parents
        if ran <= 0.2:
            mutate = creatMutate(dataval)
            child = checkIfValidChild(parents, mutate.dataval)
        else:
            child = checkIfValidChild(parents, dataval)#check if the child is one of his parents neighbors or his pearents themeself
        if child != None:
            child.myDad = parents[0]
            child.myMom = parents[1]
            return child
        counter = counter + 1
    return child

def creatMutate(dataval):#choose randomly one of the child neighbors
    neighbors = findAllOptions(dataval)
    ran = random.randint(1,len(neighbors))
    newChild = boardObject(neighbors[ran-1].dataval)
    newChild.ifMutate = True
    return newChild

def checkIfValidChild(parents, dataval):#check if the child is one of his parents neighbors or hos pearents themeself
    dadNeighbors = findAllOptions(parents[0].dataval)
    dadNeighborsDataval = []
    for board in dadNeighbors:
        dadNeighborsDataval.append(board.dataval)
    momNeighbors = findAllOptions(parents[1].dataval)
    momNeighborsDataval = []
    for board in momNeighbors:
        momNeighborsDataval.append(board.dataval)
    child = None
    if dataval in dadNeighborsDataval:
        child = boardObject(dataval)
        child.cameFrom = parents[0]
        return child
    elif dataval == parents[0].dataval:
        return parents[0]
    elif dataval in momNeighborsDataval:
        child = boardObject(dataval)
        child.cameFrom = parents[1]
    elif dataval == parents[1].dataval:
        return parents[1]
    return child

def setMyParents(myRanges):#choose 2 parents by the range
    parents = []
    for i in range(2):
        ran = random.random()
        if ran == 0:
            print("hellooooo")
        for element in myRanges:
            if ran >= element[1] and ran < element[2]:
                parents.append(element[0])
    return parents

def giveRange(currentPopulation):#I will create array the every element contain the board object, UB and LB
    theRange = []
    incremental = 0 #sum here the balance of the range
    for board in currentPopulation:
        if board.pScoreForGenetic != None:
            theRange.append((board, incremental, incremental+board.pScoreForGenetic))
            incremental = incremental + board.pScoreForGenetic
    return theRange

def giveProbability(currentPopulation):#get probability via h value
    values = getMaxAndSumValue(currentPopulation)
    badHScore = values[0]#get the bad value from the population, the biggest h score
    sigmaHValue = values[1]
    sigmaVValue = 0
    for board in currentPopulation:#set v value
        if board.hScore != math.inf:
            board.vScoreForGenetic = (badHScore - board.hScore)/sigmaHValue
            sigmaVValue = sigmaVValue + board.vScoreForGenetic
    for board in currentPopulation: #set p value
        if sigmaVValue == 0:
            sigmaVValue =1
        if board.vScoreForGenetic != None:
            board.pScoreForGenetic = board.vScoreForGenetic/sigmaVValue

def getMaxAndSumValue(currentPopulation):
    max = 0
    sum = 0
    for board in currentPopulation:
        if board.hScore != math.inf:
            if board.hScore > max and board.hScore != math.inf:
                max = board.hScore
            sum = sum + board.hScore
    arr = [max, sum]
    return arr

def printWithProbability(startingBoard, goalBoard):
    sb = boardObject(startingBoard)
    path = simulatedAnnealing(sb, goalBoard)
    if len(path) == 0:
        print("No path found.")
    else:
        counter = 1
        for board in path:
            if board.dataval == startingBoard:
                print("Board 1 (starting position):")
                print2dArray(board.dataval)
                for tuple in forPrintWithProbability:
                    step = findTheStep(board.dataval, tuple[1].dataval)
                    if len(step) == 2:
                        print("Action:" + str(step[0]) + "->" + str(step[1]) + "; probability: " + str(tuple[0]))
                    elif len(step) == 1:
                        print("Action:" + str(step[0]) + "-> out; probability: " + str(tuple[0]))
                print("-----")
            elif board.dataval == goalBoard:
                print("Board " + str(counter) + " (goal position):")
                print2dArray(board.dataval)
            else:
                print("Board " + str(counter) + ":")
                print2dArray(board.dataval)
                print("-----")
            counter += 1

def findTheStep(current, optionalStep):
    currentAgent = findAgentLocation(current, True)
    currentAgentForDel = copy.deepcopy(currentAgent)
    optionalStepAgent = findAgentLocation(optionalStep, True)
    optionalStepAgentForDel = copy.deepcopy(optionalStepAgent)
    output = []
    for point1 in currentAgent :
        for point2 in optionalStepAgent:
            if point1 == point2:
                currentAgentForDel.remove(point1)
                if len(current) == len(optionalStepAgent):
                    optionalStepAgentForDel.remove(point2)
    output.append(currentAgentForDel[0])
    if len(current) == len(optionalStepAgent):
        output.append(optionalStepAgentForDel[0])
    return output

def printWithBag (startingBoard, goalBoard):
    sb = boardObject(startingBoard)
    path = beamSearch(sb, goalBoard)
    if len(path) == 0:
        print("No path found.")
    else:
        counter = 1
        for board in path:
            if board.dataval == startingBoard:
                print("Board 1 (starting position):")
                print2dArray(board.dataval)
                prinyMyBags(board, counter, True)
            elif board.dataval == goalBoard:
                print("Board " + str(counter) + " (goal position):")
                print2dArray(board.dataval)
            else:
                print("Board " + str(counter) + ":")
                print2dArray(board.dataval)
                prinyMyBags(board, counter, False)
                print("-----")
            counter += 1

def prinyMyBags(board, counter, flag):
    print("-----")
    index = 1
    for x in board.my3TopNeighbor:
        if index == 1:
            print("Board " + str(counter+1) + "a:")
            print2dArray(x.dataval)
            if len(board.my3TopNeighbor) > 1:
                print("-----")
        if index == 2:
            print("Board " + str(counter+1) + "b:")
            print2dArray(x.dataval)
            if len(board.my3TopNeighbor) > 2:
                print("-----")
        if index == 3:
            print("Board " + str(counter+1) + "c:")
            print2dArray(x.dataval)
            if flag:
                print("-----")
        index = index + 1

def hillClimbing (startingBoard, goalBoard):
    restart = 0
    visited = []#initilize thr list of the boards that ive already been
    while restart < 6:
        current = startingBoard  #creation of the board object
        flag = True
        while flag: # while i am not go far away from the goalboard
            if goalBoard == current.dataval:
                return reconstruct_path(current)
            myNeighborPriorityQueue = queue.PriorityQueue()
            myNeighbor = findAllOptions(current.dataval)
            for board in myNeighbor: #setup all myNeighbor
                board.cameFrom = current
                board.hScore = findHeuristic(board.dataval, goalBoard)
                myNeighborPriorityQueue.put((board.hScore, board))
            neighbor = None
            while neighbor == None and not myNeighborPriorityQueue.empty():
                if not myNeighborPriorityQueue.empty():
                    temp = myNeighborPriorityQueue.get()#get the element with the min h value
                    if temp[1].dataval not in visited:#check we didnt go throw there
                        neighbor = temp[1] # #choose the neighbor with the min heuristic and not visited
            if myNeighborPriorityQueue.empty() or neighbor == None:
                break
            visited.append(neighbor.dataval)
            current.hScore = findHeuristic(current.dataval, goalBoard)
            if neighbor.hScore > current.hScore or myNeighborPriorityQueue.empty(): #if i far away from goal
                flag = False
            current = neighbor
        restart = restart + 1
    return []

forPrintWithProbability = [] #global variable for the pirnt with probability for algo SA

def simulatedAnnealing (startingBoard,goalBoard):
    current = startingBoard # initial the starting board
    current.hScore = findHeuristic(current.dataval, goalBoard)
    t = 1
    while t < 101 :
        if current.dataval == goalBoard:  # check if i reach the goal
            return reconstruct_path(current)
        T = schedule(t) #liniar function
        if T == 0 :
            if current.dataval == goalBoard:#check if i reach the goal
                return reconstruct_path(current)
            else:
                return []
        myNeighbor = findAllOptions(current.dataval)
        myNeighborWithPriority = queue.PriorityQueue()
        for neighbor in myNeighbor:
            ran = random.random()
            neighbor.probability = ran
            myNeighborWithPriority.put((-ran, neighbor))#the minus is for taking the element with the highest priority
        next = myNeighborWithPriority.get()[1]
        if current.dataval == startingBoard.dataval:
            forPrintWithProbability.append((next.probability, next))  # for the print of the steps with the probability
        next.hScore = findHeuristic(next.dataval, goalBoard)
        next.cameFrom = current
        sub = next.hScore - current.hScore#calc sub between the h scores
        if sub <= 0 :# if im in "better" place
            current = next
        else:
            randomNum = random.random()
            index = 1-(math.e)**(sub/T)
            if randomNum < index:
                current = next #change the current in the relevant approximatlly
        t += 1

def schedule(t): #the linear function
    return (100-t)

def beamSearch(startingBoard, goalBoard):
    current = startingBoard
    visited = []
    visited.append(current.dataval)
    myTop3 = []
    myTop3.append(current)
    while len(myTop3) > 0: # for every "level" in the tree
        openSet = queue.PriorityQueue()
        for board in myTop3: #expend my 3 top boards
            myNeighbor = findAllOptions(board.dataval)
            my3TopNeighborPriorityQueue = queue.PriorityQueue()
            my3TopNeighbor = []
            for neighbor in myNeighbor:
                neighbor.cameFrom = board
                neighbor.hScore = findHeuristic(neighbor.dataval, goalBoard)
                if neighbor.dataval == goalBoard:
                    my3TopNeighborPriorityQueue.put((neighbor.hScore, neighbor))
                    for i in range(3):
                        if my3TopNeighborPriorityQueue.empty():
                            break
                        temp = my3TopNeighborPriorityQueue.get()[1]
                        if temp.dataval not in my3TopNeighbor:
                            my3TopNeighbor.append(temp)
                    board.my3TopNeighbor = my3TopNeighbor
                    return reconstruct_path(neighbor)
                openSet.put((neighbor.hScore, neighbor))
                my3TopNeighborPriorityQueue.put((neighbor.hScore, neighbor))
            for i in range(3):
                if my3TopNeighborPriorityQueue.empty():
                    break
                temp = my3TopNeighborPriorityQueue.get()[1]
                if temp.dataval not in my3TopNeighbor:
                   my3TopNeighbor.append(temp)
            board.my3TopNeighbor = my3TopNeighbor
        myTop3 = []
        for i in range(3): # choose the best top 3 from all the level
            if openSet.empty():
                break
            temp = openSet.get()[1]
            if temp.dataval not in visited: # to make sure im not going back
                myTop3.append(temp)
                visited.append(temp.dataval)
    return []

def printWithHeuristic(startingBoard,goalBoard):
    sb = boardObject(startingBoard)
    path = A_Star(sb, goalBoard)
    if len(path) == 0:
        print("No path found.")
    else:
        counter = 1
        for board in path:
            if board.dataval == startingBoard:
                print("Board 1 (starting position):")
                print2dArray(board.dataval)
            elif board.dataval == goalBoard:
                print("Board " + str(counter) + " (goal position):")
                print2dArray(board.dataval)
                print("Heuristic: " + str(board.hScore))
            else:
                print("Board " + str(counter) + ":")
                print2dArray(board.dataval)
                print("Heuristic: " + str(board.hScore))
                print("-----")
            counter += 1

def printWithNoHeuristic(startingBoard,goalBoard,algoNum):
    sb = boardObject(startingBoard)
    path = None
    if algoNum == 1:
        path = A_Star(sb, goalBoard)
    elif algoNum == 2:
        path = hillClimbing(sb, goalBoard)
    elif algoNum == 3:
        path = simulatedAnnealing(sb, goalBoard)
    elif algoNum == 4:
        path = beamSearch(sb, goalBoard)
    elif algoNum == 5:
        path = myGenetic(sb, goalBoard)
    if len(path) == 0:
        print("No path found.")
    else:
        if len(path) == 1 and algoNum == 5:
            print("No path found.")
        else:
            counter = 1
            for board in path:
                if board.dataval == startingBoard:
                    print("Board 1 (starting position):")
                    print2dArray(board.dataval)
                elif board.dataval == goalBoard:
                    print("Board " + str(counter) + " (goal position):")
                    print2dArray(board.dataval)
                else:
                    print("Board " + str(counter) + ":")
                    print2dArray(board.dataval)
                    print("-----")
                counter += 1

def reconstruct_path(current): # recursive function that return a list of the path
    if current is not None:
        return reconstruct_path(current.cameFrom) + [current]
    else:
        return []

def A_Star(startingBoard, goalBoard):
    visited = []
    oppnSetOnlyVal = []# only for the value of the 2darray that present the board
    openSet = queue.PriorityQueue()
    startingBoard.gScore = 0
    startingBoard.hScore = findHeuristic(startingBoard.dataval, goalBoard)
    startingBoard.fScore = startingBoard.hScore # because the g of the start board is 0
    openSet.put((startingBoard.fScore, startingBoard.hScore, startingBoard))#add to the PriorityQueue the starting board element
    oppnSetOnlyVal.append(startingBoard.dataval)
    while openSet.qsize() > 0 and len(visited) <= len(startingBoard.dataval)*len(startingBoard.dataval[0]*100):
        temp = openSet.get()# get the elemnt with the lowest f score (if there more than 1 element with the same f score it will tkae the lowest h score between them)
        current = temp[2]# every elemnt include tuple of 3 elements
        oppnSetOnlyVal.remove(current.dataval)
        visited.append(current.dataval)
        if ifEquals(current.dataval, goalBoard):
            return reconstruct_path(current)
        myNeighbor = findAllOptions(current.dataval)
        for neighbor in myNeighbor:
            if not ifInList(neighbor.dataval, oppnSetOnlyVal) and not ifInList(neighbor.dataval, visited):
                neighbor.gScore = current.gScore + 1
                neighbor.cameFrom = current
                neighbor.hScore = findHeuristic(neighbor.dataval, goalBoard)
                neighbor.fScore = neighbor.gScore + neighbor.hScore
                openSet.put((neighbor.fScore, neighbor.hScore, neighbor))
                oppnSetOnlyVal.append(neighbor.dataval)
            elif ifInList(neighbor.dataval, oppnSetOnlyVal):
                neighbor.gScore = current.gScore + 1
                if neighbor.gScore < getTheEqualObjectFromOpenSet(openSet, neighbor.dataval).gScore:
                    neighbor.cameFrom = current
                    neighbor.hScore = findHeuristic(neighbor.dataval, goalBoard)
                    neighbor.fScore = neighbor.gScore + neighbor.hScore
                    openSet = swapTheObjects(openSet, neighbor)
    return []

def getTheEqualObjectFromOpenSet (openSet, neighbor2Darray):#retirn the object that hava the same dataval as neighbor
    tempList = list(openSet.queue)
    for x in tempList:
        if ifEquals(x[2].dataval, neighbor2Darray):
            return x[2]

def swapTheObjects(openSet, neighbor):# swap the objects with the same dataval but the g score of the neighbor lower then the same dataval in the openset
    tempList = list(openSet.queue)
    for x in tempList:
        if ifEquals(x[2].dataval, neighbor.dataval):
            tempIndex = tempList.index(x)
            del tempList[tempIndex]
            tempList.append((neighbor.fScore, neighbor.hScore, neighbor))
            tempPriorityQueue = queue.PriorityQueue()
            for y in tempList:
                tempPriorityQueue.put(y)
            return tempPriorityQueue

def ifInList(twoDarray, listOfArrays):
    for x in listOfArrays:
        if ifEquals(x, twoDarray):
            return True
    return False

def ifEquals(array1, array2):
    for i in range(len(array1)):
        for j in range(len(array1[0])):
            if array1[i][j] != array2[i][j]:
                return False
    return True

def findHeuristic(optionBoard, goalBoard): # find the value by absoulot distance between the agents in the current board to the agents in the goal boardי
    targets = findAgentLocation(goalBoard, False)# return tuple of pair index of the agents
    current = findAgentLocation(optionBoard, False)
    heuristicValue = 0
    if len(targets) < len(current):
        numOfDropAgent = len(current)-len(targets)
        for x in range(numOfDropAgent):
            targets.append((0,len(optionBoard))) # add new "targets"
    if len(current) < len(targets) or len(current) == 0:
        return math.inf
    for m in current:
        tempForDis = []  # list of the distance of the agent from every target
        for n in targets:
            if n[1] == len(optionBoard):
                tempNum = (abs(m[0] - len(optionBoard)))
            else:
                tempNum = abs(m[0]-n[0]) + abs((m[1]-n[1]))
            tempForDis.append(tempNum)
        heuristicValue = heuristicValue + min(tempForDis)# Add the min value for the whole heuristic
        minIndex = tempForDis.index(min(tempForDis)) #delete the "match" variable
        del targets[minIndex]
    return heuristicValue

def findAllOptions(startingBoard):
    optionsList = []
    for i in range(len(startingBoard)):
        for j in range(len(startingBoard[0])):
            if startingBoard[i][j] == 2: # add all the valid neibhor of that board
                if i > 0 and startingBoard[i-1][j] == 0:
                    dataval = copy.deepcopy(startingBoard)
                    dataval[i-1][j] = 2
                    dataval[i][j] = 0
                    tempBoard = boardObject(dataval)
                    optionsList.append(tempBoard)
                if i < len(startingBoard)-1 and startingBoard[i+1][j] == 0:
                    dataval = copy.deepcopy(startingBoard)
                    dataval[i+1][j] = 2
                    dataval[i][j] = 0
                    tempBoard = boardObject(dataval)
                    optionsList.append(tempBoard)
                if j > 0 and startingBoard[i][j-1] == 0:
                    dataval = copy.deepcopy(startingBoard)
                    dataval[i][j-1] = 2
                    dataval[i][j] = 0
                    tempBoard = boardObject(dataval)
                    optionsList.append(tempBoard)
                if j < len(startingBoard)-1 and startingBoard[i][j+1] == 0:
                    dataval = copy.deepcopy(startingBoard)
                    dataval[i][j+1] = 2
                    dataval[i][j] = 0
                    tempBoard = boardObject(dataval)
                    optionsList.append(tempBoard)
                if i == len(startingBoard)-1:
                    dataval = copy.deepcopy(startingBoard)
                    dataval[i][j] = 0
                    tempBoard = boardObject(dataval)
                    optionsList.append(tempBoard)
    return optionsList

def findAgentLocation(Board, flag): #binary variable for if we need to do +1 for the index for the print of SA
    myTarget = []
    for i in range(len(Board)):
        for j in range(len(Board[0])):
            if Board[i][j] == 2:
                if flag:
                    myTarget.append((i+1, j+1))
                if not flag:
                    myTarget.append((i, j))
    return myTarget

def print2dArray(theArray):
    print ('  1 2 3 4 5 6')
    for i, r in enumerate(theArray):
        print(f"{i+1}:", end='')
        for c in r:
            if c == 0:
                print("  ", end='')
            elif c == 1:
                print("@ ", end='')
            elif c == 2:
                print ("* ", end='')
        print()


