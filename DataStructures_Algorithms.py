import datetime

CurrentDate = datetime.datetime.now()
#format(year, month, day, hour, minute, second)
StartDate = datetime.datetime(2021, 10, 5, 22, 5, 57)
NewDate = CurrentDate - StartDate 

print("I want to learn Data Structures and Algorithms")
print ("This code has being written for", NewDate.days, "days now")
print (CurrentDate.strftime("""And you're viewing this live at %H:%M:%S %p. %d %B, %Y\n
Enjoy!!\n"""))


#I want to learn Data Structures and Algorithms
#Data structures: Data structures are building blocks of any software application


#Big O Notation: This is used to measure how the running time or space requirements for your program grows with respect to the input size. This is the highest value of n

#Time Complexity: This is the measure of time growth, n = number of times 
#When the running time is relative to the size of the computation i.e. the running time increases linearly, then the time complexity is order of n --> O(n)
#When the running time isn't that relative to the size the computation (which most times has to do with the index value of a list) i.e. the running time is almost constant, then the time complexity is order of 1 --> O(1)
#When the number of iterations of a certain computation is in a nested loop, then the time complexity is in order of n_square --> O(n_square)

#Space Complexity: This is the measure of space growth, this is a better complexity. k = O(log <sub 2 /> n)
#Using binary search method on a list to find a certain number will have a space complexity of order of log n --> O(log n)

#Array: Static array doesn't allow you to add new items, dynamic array(list) gives you that priviledge
"""
print("ARRAY \n")

age = [12, 21, 43, 21, 35, 15]
print(age[2]) #This has a time complexity of O(1)
for i in range(len(age)): #This has a time complexity of O(n)
    if age[i] == 15:
        print(i)
for x in age: #This has a time complexity of O(n)
    print(x)
age.insert(1, 71) #1 is the index and 71 is the age to be inserted, this has a time complexity of O(n)
age.remove(71) #71 is the item to be deleted, this has a time complexity of O(n)

print("\nEXERCISE ARRAY \n")

expense = [
     {"January": 2200,
     "February": 2350,
     "March": 2600,
     "April": 2130,
     "May": 2190}
    ]
#This is how to  call a dict within a list, The 0(position of the dict in the list), values(the values in the dict) and 1(position of the value in the dict)
values_ = list(expense[0].values())[1]  
price = expense[0].values() #This stores all prices in a list
extraExp = expense[0]["February"] - expense[0]["January"] #This calculate the extra price spent in feb compared to jan
priceN = list(price)
print(sum(priceN[0:3])) #This gives the sum of the first 3 month of the year
for i in range(len(priceN)): #This checks if a certain is in the list
    if (priceN[i] == 2000) == True:
        print(priceN[i], "has be found")
expense[0]["June"] = 1980 #This adds a new item to the list
expense[0]["April"] -= 200 #This sets reduces the number of a certain item in the dict
print(expense)

print("\n")
"""


#Linked List: This stores the current data and address of the next data in a memory node, This thing is hard, God pls help me
#Double Linked List: This stores the address of the previous data, the current data and address of the next data in a memory node
"""
print("LINKED LIST \n")

class Node:
    def __init__(self, data = None, next = None): #This declares the parameters to be passed in
        self.data = data #This declares the data
        self.next = next #This declares the next data
class LinkedList:
    def __init__(self):
        self.head = None
    def insertBeginning(self, data): #This declares a function to insert a number at the beginning
        node = Node(data, self.head) #This calls the node class, declares data and self.head in place of next as the various arguments
        self.head = node
    def print(self):
        if self.head is None: #This checks if the linked list parameter is empty
            print("Linked list is empty")
            return #This ends the if statement

        itr = self.head
        llstr = " "
        while itr: #While itr exist it runs the below code
            llstr += str(itr.data) + "-->" #This adds the data to the variable
            itr = itr.next #This calls the next arguments inputted
        print(llstr)
    def insertEnd(self, data): #This declares a function to insert a number at the end
        if self.head is None:
            self.head = Node(data, None)
            return
        itr = self.head
        while itr.next:
            itr = itr.next
        itr.next = Node(data, None)
    def insertValues(self, data_list):
        self.head = None
        for data in data_list: #This loops through the list and creates a linked list
            self.insertEnd(data)
    def getLength(self): #This gets the length of the list
        count = 0 
        itr = self.head
        while itr: #This counts through the list
            count += 1
            itr = itr.next
        return count
    def removeAt(self, index): #This removes a certain item from the linked list
        if index < 0 or index >= self.getLength(): #This checks if the inputted index is within the range of the list
            raise Exception("Invalid index")
        if index == 0:
            self.head = self.head.next
            return
        count = 0 
        itr = self.head
        while itr: #This loops through the list
            if count == index - 1:
                itr.next = itr.next.next
                break
            itr = itr.next
            count += 1
    def insertAt(self, index, data): #This inserts a certain item in a certain position in the linked list
        if index < 0 or index >= self.getLength(): #This checks if the inputted index is within the range of the list
            raise Exception("Invalid index")
        if index == 0:
            self.insertBeginning(data)
            return
        count = 0 
        itr = self.head
        while itr: #This loops through the list
            if count == index - 1:
                node = Node(data, itr.next)
                itr.next = node
                break
            itr = itr.next
            count += 1
      
    #Exercise  #Unfinished business --------------------------------------------------------------------------------------------------------------------->
    #def insertAfterValue(self, data_after, dataInserted):
    #    count = 0
    #    itr = self.head
    #    while itr:
    #        #if 
    #        node = Node(data_after, dataInserted)
    #        itr.next = node 

    #       #if data_after == itr.next:
    #        print(itr.next)
    #        count += 1
            
 
if __name__ == "__main__":
    ll = LinkedList()
    ll.insertBeginning(5)
    ll.insertBeginning(71)
    ll.insertEnd(26)
    ll.print()
    ll.insertValues(["Banana", "Cherry", "Mango", "Apple"])
    ll.removeAt(2)
    ll.insertAt(2, "Figs") #2 is the index number and Figs is the item
    ll.print()
    print("lenght:", ll.getLength()) 
    print('\n')
    #ll.insertAfterValue("Cherry", 2)
"""



#Hash Table: Dicitionary data structure in python uses hash table concept for functionality
"""
print("HASH TABLE")

stockPrice = {} #This is the best way to print all the list contents from a file, especially when dealing with large datas
with open("Hash_Table1.csv", "r") as f:
    for line in f:
        tokens = line.split(",") #This sets the demacator as ,
        day = tokens[0] #This fits the day in an array
        price = float(tokens[1]) #This fits the prices in an array
        stockPrice[day] = price #This appends the various values
print(stockPrice)
print(stockPrice["9-Mar"]) #This finds the price for a particular date, this can be performed better with dictionary

#Implementation
def getHash(key): #This is a function to get hash value of a value
    val = 0
    for char in key: #This determines the value to be passed in as a char
        val += ord(char) #The ord fucntion gives the ascii equivalent of the char passed in
    return val % 100
print(getHash("march 9"))

#Unfinised business --------------------------------------------------------------------------------------------------------------------------------->

print("\n")
"""

#Queue: It has both enqueue and deque functions.. It's implemented using set
 

from collections import deque
#Stack: This is a data structure that follows the LIFO order, Deque method is best used to implement a stack
"""
print("STACK")

stack = deque()
#for dirs in dir(stack): #This dir function is like /? in cmd
#    print(dirs)
stack.append(1)
stack.append(2)
stack.append(3)
stack.append(4)
stack.pop()
print(stack)

#Implementing stack in python
class stack_:
    def __init__(self):
        self.container = deque()
    def push(self, val): #This appends a value to the stack
        self.container.append(val)
        
    def pop(self): #This deletes a value from the stack
        self.container.pop()
    def peek(self): #This returns the last value in the container
        return self.container[-1]
    def isEmpty(self): #This checks if the container is empty
        if len(self.container) == 0:
            print("Is empty")
    def size(self): #This checks for the size of the container
        return len(self.container)

s = stack_()
s.push(1)
s.push(2)
s.push(3)
s.push(4)
s.pop()

#Exercise ------------------------------------------------------------------------------------------------------------------------------------>
#word = "Favour syre"
#for char in word:
#    print(word[-1])

print("\n")
"""


#Trees: This is an hierachical data structure

#General Tree
"""
class treeNode: 
    def __init__(self, data):
        self.data = data
        self.children = [] 
        self.parent = None #The parent function is declared as none because it's the parent
    def addChild(self, child):
        child.parent = self
        self.children.append(child)
    def getLevel(self): #This is to get the level of the various items
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level
    def printTree(self):
        spaces = " " * self.getLevel() * 3 #This gives spaces depending on the node's level
        prefix = spaces + "|__" if self.parent else "" #This gives the tree a nicer format
        print(prefix + self.data)
        if self.children: #This checks if self children exist
            for child in self.children:
                child.printTree()

def buildTree():
    root = treeNode("Electronics")

    laptop = treeNode("Laptop")
    laptop.addChild(treeNode("Mac"))
    laptop.addChild(treeNode("Microsoft"))
    laptop.addChild(treeNode("Samsung"))
    root.addChild(laptop)

    phone = treeNode("Phone")
    phone.addChild(treeNode("iPhone"))
    phone.addChild(treeNode("Xaomi"))
    phone.addChild(treeNode("Samsung"))
    root.addChild(phone)

    tv = treeNode("TV")
    tv.addChild(treeNode("Apple"))
    tv.addChild(treeNode("Hisense"))
    tv.addChild(treeNode("Samsung"))
    root.addChild(tv)

    return root

if __name__ == "__main__":
    root = buildTree()
    root.printTree()
    #print(root.level)

print("\nExercise \n") #Unfinished business ---------------------------------------------------------------------------------------------------------->

class companyTree:
    def __init__(self, data):
        self.data = data
        self.children = [] 
        self.parent = None #The parent function is declared as none because it's the parent
    def addChild(self, child):
        child.parent = self
        self.children.append(child)
    def getLevel(self): #This is to get the level of the various items
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level
    def printTree(self):
        spaces = " " * self.getLevel() * 3 #This gives spaces depending on the node's level
        prefix = spaces + "|__" if self.parent else "" #This gives the tree a nicer format
        print(prefix + self.data)
        if self.children: #This checks if self children exist
            for child in self.children:
                child.printTree()

def buildCompany():
    root = companyTree("Uchiha Syre (CEO)")

    dpt1 = companyTree("Marketing")
    dpt1.addChild(companyTree("Syre Shelby (Marketing Head)"))
    root.addChild(dpt1)

    dpt2 = companyTree("Production")
    dpt2.addChild(companyTree("Syre Musk (Production)"))
    root.addChild(dpt2)

    return root

if __name__ == "__main__":
    root = buildCompany()
    root.printTree()
    #print(root.level)
print("\n")
"""


#Binary Tree: Could be inorder, preorder or postorder. The root node determines the position and is implemented using set
"""
class binaryTree(): #This declares the main class for the binary tree
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    def addChild_(self, data): #This adds branches to either the left or right
        if data == self.data:
            return 
        if data < self.data:
            if self.left:
                self.left.addChild_(data)
            else:
                self.left = binaryTree(data)
        else:
            if self.right:
                self.right.addChild_(data)
            else:
                self.right = binaryTree(data)
    def inOrderTransversal(self): #This sorts the tree in an in-order form (smallest --> largest)
        elements = []
        if self.left:
            elements += self.left.inOrderTransversal()
        elements.append(self.data)
        if self.right:
            elements += self.right.inOrderTransversal()
        return elements

def buildTree(elements):
    root = binaryTree(elements[0])
    for i in range(1, len(elements)):
        root.addChild_(elements[i])
    return root

    #Unfinished business --------------------------------------------------------------------------------------------------------------->
    #def search(self, val): #This function searches for a value
    #    if self.data == val:
    #        return Tree
    #    if val < self.data:
    #        if self.left:
    #            return self.left.search(val)
    #        else:
    #            return False
    #    else:
    #        if self.right:
    #            return self.right.search(val)
    #        else:
    #            return False


if __name__ == "__main__":
    num = [17, 11, 3, 54, 21, 52, 71, 1, 11]
    numTree = buildTree(num)
    print(numTree.inOrderTransversal())

print("\n")
"""


#Graph: This data structure is used to depict network and could either be directed or undirected
"""
class Graph:
    def __init__(self, edges):
        self.edges = edges 
        self.graphDict = {}
        for start, end in self.edges: #This loops through the data(edges) given
            if start in self.graphDict:
                self.graphDict[start].append(end) #This checks if a keyword has been repeated and then appends the value as a list
            else:
                self.graphDict[start] = [end]
        print("Graph dict: ", self.graphDict)

if __name__ == "__main__":
    routes = [
        ("Lagos", "Paris"),
        ("Lagos", "London"),
        ("Paris", "London"),
        ("Paris", "New York"),
        ("London", "New York"),
        ("New York", "Toronto"),
        ("New York", "Cape Town")
    ]
    #Exercise ----------------------------------------------------------------------------------------------------------------------------->
    routeGraph = Graph(routes)

print("\n")
"""

from timeit import default_timer as timer
from test import timeSizeCheck #timeCheck, sizeCheck #This imports the time/size check decorator from the other file
import sys
#Binary search: This has better time complexity than linear search
"""
@timeSizeCheck
def linearSearch(numList, numFind): #This is a function to perform a linear search
    for index, element in enumerate(numList): #Enumerate function tuples the index and element in a list
        if element == numFind: #if the element equals the number to be found it returns the index
            return index
    return -1 #This means that the number wasn't found in the list

@timeSizeCheck
def binarySearch(numList, numFind): #This is a function to perform a binary search
    leftIndex = 0 #Beginning index is 0
    rightIndex = len(numList) - 1 #This represents the last index in the list
    midIndex = 0 #Once the other half has been discarded, the mid element then automatically becomes the first index

    while leftIndex <= rightIndex: #This loops through until the number to be found is found
        midIndex = (leftIndex + rightIndex) // 2 #This returns the mid index (Example: 5 // 2 = 2 instead of 2.5)
        midNum = numList[midIndex] #This assigns the mid index as the middle number
        if midNum == numFind: #This checks if the mid number equals the number to be found and returns it else -->
            return midIndex
        if midNum < numFind: #If the number to be found is greater than the middle number, the new left index gets effected
            leftIndex = midIndex + 1
        else:
            rightIndex = midIndex - 1
    return

if __name__ == "__main__":
    numList = [8, 12, 25, 31, 42, 63, 67, 91, 101]
    numFind = 25
    index = binarySearch(numList, numFind)
    index = linearSearch(numList, numFind)
    if index:
        print(f"Number found at index {index} using binary search")
    else:
        print("Number not found")
    print("\n")

    numList = [i for i in range(1000010)]
    numFind = 10000
    index = binarySearch(numList, numFind)
    index = linearSearch(numList, numFind)


#Exercise ----------------------------------------------------------------------------------------------------------------->
    print("\n")
    numList = [0, 1, 2, 3, 3, 3, 6, 7, 8, 9]
    numFind = 3
    index = binarySearch(numList, numFind)

    print(f"Number found at index {index} using binary search")

print("\n")
"""


#Bubble Sort:  This sorts by comparing the first two numbers and rearranging it as it loops through the list
"""
@timeSizeCheck
def bubbleSort(elements): #A function for bubble sort computation
    size = len(elements) #This sets the size as the length of the list to be sorted
    for i in range(size - 1): #This loops through the function
        sorted = False #This checks if the elements is sorted already in other to make it more efficient
        for j in range(size - 1 - i): #This reprensents the index
            if elements[j] > elements[j + 1]: #This compares the various elements to make sure it's sorted
                elements[j], elements[j + 1] = elements[j + 1], elements[j] #This swaps the items in the list
                sorted = True #This declares the swapped state to be true
        if sorted == False: #If there was no elements sorted through the first loop then it means that the elements is already sorted, therefore no need to loop twice and then it breaks
            break 

if __name__ == "__main__":
    elements = [5, 7, 8, 17, 81, 100, 25]
    #elements = ["ele", "ada", "yes", "ice"]
    #elements = [1, 2, 3]
    bubbleSort(elements)
    print(elements)

#You have an unfinished practice exercise to attend to
print("\n")
"""


#Quick Sort: Using lomuto partition, Given [1, 2, 3, 4, 5]; 1 is the pivot, 2 is start and 5 is end.. Every elements is compared with the pivot as it loops through the elements
"""
def swap(a, b, arr): #This function handles the swapping of elements
    if a != b:
        arr[a], arr[b] = arr[b], arr[a]
def partition(elements, start, end): #This function handles the partition
    pivotIndex = start
    pivot = elements[pivotIndex]

    while start < end:
        while start < len(elements) and elements[start] <= pivot:
            start += 1
        while elements[end] > pivot:
            end -= 1
        if start < end:
            swap(start, end, elements)
    swap(pivotIndex, end, elements)
    return end
def quickSort(elements, start, end): #This function handles the quick sorting of the elements
    if start < end:
        pi = partition(elements, start, end)
        quickSort(elements, start, pi - 1)
        quickSort(elements, pi + 1, end)

if __name__ == "__main__":
    elements = [11, 9, 29, 7, 2, 15, 28]
    quickSort(elements, 0, len(elements) - 1)
    print(elements)

print("\n")
"""


#Insertion sort: Given [2, 3, 4, 1, 3], 2 is the start and 3 is the pointer then you compare and move the pointer once its sorted
#worstCasePerformance: O(nsquare); bestCasePerformance: O(n) and averagePerformance: O(nsquare)
"""
def insertionSort(elements): #This declares a function for insertion sorting
    for i in range(1, len(elements)): #This starts looping from the 2nd indexed number to the last and leaves the first indexed number
        anchor = elements[i] #This sets the number of the indexed number before the sorted part of the list as the anchor
        j = i - 1 
        while j >= 0 and anchor < elements[j]: #This checks if the preceeding number is greater than the later
            elements[j + 1] = elements[j] #If its greater it then swaps the number 
            j = j - 1
        elements [j + 1] = anchor #if the preceeding number isn't greater than the first, it then sets the first number as the anchor

if __name__ == "__main__":
    elements = [11, 9, 29, 7, 2, 15, 28]
    insertionSort(elements) #This calls the insertion function on the the elements
    print(elements)

print("\nEXERCISE \n") 
#Unfinished practical exercise ------------------------------------------------------------------------------------------------------------------->
"""


#Merge Sort(Divide and Conquer): Given [1,2,3] and [2,5,6], you compare each elements of the various arrays and then sort them in a general array or
#Given [1,4,5,6,2,0], you break them into smaller bits of array([1,4], [5, 6] & [2,0]) and then start sorting them uniquely until you get a general sorted array
"""
def mergeSort(arr): #This declares a function to handle merge sort technique
    if len(arr) <= 1: #This checks if it's a single array and returns it as it's already sorted
        return 
    mid = len(arr) // 2
    left = arr[:mid] #This slice the array and sets the left side of the array according to the specified slicing as the left array
    right = arr[mid:]
    mergeSort(left)             
    mergeSort(right) #This merge sorts the various divided arrays
    mergeTwoSort(left, right, arr) #This merges the two sorted arrays

def mergeTwoSort(a, b, arr): #This declares a function that merges two already single sorted arrays
    lenA = len(a)
    lenB = len(b)
    i = j = k = 0
    while i < lenA and j < lenB:
        if a[i] <= b[j]:
            arr[k] = a[i]
            i += 1
        else:
            arr[k] = b[j]
            j += 1
        k += 1
    while i < lenA:
        arr[k] = a[i]
        i += 1
        k += 1
    while j < lenB:
        arr[k] = b[j]
        j += 1
        k += 1

if __name__ == "__main__":
    arr = [10, 3, 15, 7, 8, 23, 98, 29] 
    mergeSort(arr)
    print(arr)
#Unfinished business ------------------------------------------------------------------------------------------------------------------>
"""


#Shell Sort: This an optimized form of selection sort and starts with a gap, sorts and then reduce the gap as it loops through the array
"""
def shellSort(arr): #A function to handle the shell sort technique
    size = len(arr)
    gap = size // 2 #This sets the gap of the sorting algorithm
    while gap > 0: #While gap is greater than zero the following computations are done
        for i in range(gap, size):
            anchor = arr[i] #This sets the anchor
            j = i
            while j >= gap and arr[j - gap] > anchor: #While anchor is >= gap and the number to be compared is greater than the anchor, the following tasks takes place
                arr[j] = arr[j - gap] #This swaps the elements
                j -= gap
            arr[j] = anchor
        gap = gap // 2

if __name__ == "__main__":
    elements = [21, 38, 29, 17, 4, 25, 11, 32, 9]
    shellSort(elements)
    print(elements)
#Unfinished exercsie ------------------------------------------------------------------------------------------------>
"""


#Selection Sort: This sets a pointer on the first element, then checks for the minimum value in the other elements swaps it and lookout for the new minimum number
#by looping through until all the arrays has been sorted
"""
def minElement(arr): #A function to handle finding minimum element in an array
    min = 100000000
    for i in range(len(arr)):
        if arr[i] < min:
            min = arr[i]
    return min

def selectionSort(arr): #This funciton handles the selection sorting technique
    size = len(arr)
    for i in range(size - 1): #This loops through the array
        minIndex = i
        for j in range(minIndex + 1, size):
            if arr[j] < arr[minIndex]: #This compares the number and checks the one that's greater
                minIndex = j
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i] #This handles the swapping

if __name__ == "__main__":
    elements = [21, 38, 29, 17, 4, 25, 11, 32, 9]
    selectionSort(elements)
    print(elements)
    print(minElement(elements))
    print(min(elements))
    #print(elements) 
print("\n")
"""



#Binary search, linked lists and complexity 
"""
Question1: Alice has some cards with numbers written. She arranges the cards in decreasing order and lays them out face down
in a sequence on a table. She challenges Bob to pick out a card containing a given number by turning over as few cards as possible. 
Write down a function to help Bob locate the card
[13, 11, 12, [7], 4, 3, 1, 0]

cards = list of numbers sorted in decresing order
query = number to be found
position = position of query


#Based on the above variables we create a signature of our function
def locate_cards(cards, query):
    pass

#Next is to come up with some examples of inputs and outputs
cards = [13, 11, 12, 7, 4, 3, 1, 0]
query = 7
position = 3
result = locate_cards(cards, query)
print (result) #None because there's nothing in the function
print (result == position) #False because they don't equal

#tst = [13, 11, 12, 7, 4, 3, 1, 0]
#for i in tst:
#    if i == 7:
#        print("Number has been found in")
"""

#Theres more to be learnt and attended to as learning doesn't stop
#Have to go on and learn some other stuffs before coming back to attend to dynamic programming and the unfinished exercises, hopefully I gain real experience before I get back

