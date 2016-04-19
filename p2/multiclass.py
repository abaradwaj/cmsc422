from binary import *
from util import *
from numpy import *

class OAA:
    def __init__(self, K, mkClassifier):
        self.f = []
        self.K = K
        for k in range(K):
            self.f.append(mkClassifier())

    def train(self, X, Y):
        for k in range(self.K):
            print 'training classifier for', k, 'versus rest'
            Yk = 2 * (Y == k) - 1   # +1 if it's k, -1 if it's not k
            try:
                self.f[k].fit(X, Yk) # For sklearn implementations
            except:
                self.f[k].train(X, Yk) # For implementations of binary.py

    def predict(self, X, useZeroOne=False):
        vote = zeros((self.K,))
        for k in range(self.K):
            probs = self.f[k].predict_proba(X.reshape(1,-1))
            if useZeroOne:
                vote[k] += 1 if probs[0,1] > 0.5 else 0
            else:
                vote[k] += probs[0,1]   # weighted vote
        return argmax(vote)

    def predictAll(self, X, useZeroOne=False):
        N,D = X.shape
        Y   = zeros(N, dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:], useZeroOne)
        return Y
        

class AVA:
    def __init__(self, K, mkClassifier):
        self.f = []
        self.K = K
        for i in range(K):
            self.f.append([])
        for j in range(K):
            for i in range(j):
                self.f[j].append(mkClassifier())

    def train(self, X, Y):
        for i in range(self.K):
            for j in range(i):
                print 'training classifier for', i, 'versus', j
                Xij =  X[ (Y==i) | (Y==j), : ]
                Yij = (Y[ (Y==i) | (Y==j) ] == j) * 2 - 1 # +1 if it's j, -1 if it's i
                try:
                    self.f[i][j].fit(Xij, Yij) # For sklearn implementations
                except:
                    self.f[i][j].train(Xij, Yij) # For implementations of binary.py

    def predict(self, X, useZeroOne=False):
        vote = zeros((self.K,))
        for i in range(self.K):
            for j in range(i):
                probs = self.f[i][j].predict_proba(X.reshape(1,-1))
                if useZeroOne:
                    p = 1 if probs[0,1] > 0.5 else 0
                    vote[j] += p
                    vote[i] -= p
                else:
                    vote[j] += probs[0,1]   # weighted vote
                    vote[i] -= probs[0,1]   # weighted vote
        return argmax(vote)

    def predictAll(self, X, useZeroOne=False):
        N,D = X.shape
        Y   = zeros((N,), dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:], useZeroOne)
        return Y
        
    
class TreeNode:
    def __init__(self):
        self.isLeaf = True
        self.label  = 0
        self.info   = None

    def setLeafLabel(self, label):
        self.isLeaf = True
        self.label  = label

    def setChildren(self, left, right):
        self.isLeaf = False
        self.left   = left
        self.right  = right

    def isLeaf(self): return self.isLeaf
    
    def getLabel(self):
        if self.isLeaf: return self.label
        else: raise Exception("called getLabel on an internal node!")
        
    def getLeft(self):
        if self.isLeaf: raise Exception("called getLeft on a leaf!")
        else: return self.left
        
    def getRight(self):
        if self.isLeaf: raise Exception("called getRight on a leaf!")
        else: return self.right

    def setNodeInfo(self, info):
        self.info = info

    def getNodeInfo(self): return self.info

    def iterAllLabels(self):
        if self.isLeaf:
            yield self.label
        else:
            for l in self.left.iterAllLabels():
                yield l
            for l in self.right.iterAllLabels():
                yield l

    def iterNodes(self):
        yield self
        if not self.isLeaf:
            for n in self.left.iterNodes():
                yield n
            for n in self.right.iterNodes():
                yield n

    def __repr__(self):
        if self.isLeaf:
            return str(self.label)
        l = repr(self.left)
        r = repr(self.right)
        return '[' + l + ' ' + r + ']'
            

def makeBalancedTree(allK):
    if len(allK) == 0:
        raise Exception("makeBalancedTree: cannot make a tree of 0 classes")

    tree = TreeNode()
    
    if len(allK) == 1:
        tree.setLeafLabel(allK[0])
    else:
        split  = len(allK)/2
        leftK  = allK[0:split]
        rightK = allK[split:]
        leftT  = makeBalancedTree(leftK)
        rightT = makeBalancedTree(rightK)
        tree.setChildren(leftT, rightT)

    return tree

class MCTree:
    def __init__(self, tree, mkClassifier):
        self.f = []
        self.tree = tree
        for n in self.tree.iterNodes():
            n.setNodeInfo(   mkClassifier()  )

    def train(self, X, Y):
        for n in self.tree.iterNodes():
            if n.isLeaf:   # don't need to do any training on leaves!
                continue

            # otherwise we're an internal node
            leftLabels  = list(n.getLeft().iterAllLabels())
            rightLabels = list(n.getRight().iterAllLabels())

            print 'training classifier for', leftLabels, 'versus', rightLabels

            # compute the training data, store in thisX, thisY
            ### TODO: YOUR CODE HERE
            #util.raiseNotDefined()
            
            leftY = (Y == leftLabels[0])
            for l in range(len(leftLabels)):
                leftY = logical_or(leftY, Y == leftLabels[l])
            
            rightY = (Y == rightLabels[0])
            for r in range(len(rightLabels)):
                rightY = logical_or(rightY, Y == rightLabels[r])
            
            combined = logical_or(leftY, rightY)
            
            #bug right here
            thisX = X[combined]
            thisY = Y[combined]
            #here
            
            temp = (thisY == leftLabels[0])
            for l in range(len(leftLabels)):
                temp = logical_or(temp, thisY == leftLabels[l])
            
            thisY = (2 * temp) - 1
            
            try:
                n.getNodeInfo().fit(thisX, thisY) # For sklearn implementations
            except:
                n.getNodeInfo().train(thisX, thisY) # For implementations of binary.py

    def predict(self, X):
        ### TODO: YOUR CODE HERE
        #print "entered the predict function, outside of predictAll"
        curNode = self.tree
        while curNode.isLeaf == False:
            #found the bug: make sure you are referring to curNode, not self.tree
            tmp = curNode.getNodeInfo()
            pr = tmp.predict_proba(X.reshape(1,-1))
            if pr[0,1] > 0.5:
                curNode = curNode.getLeft()
            else:
                curNode = curNode.getRight()
        
        #print 'finished while loop inside predict'        
        if curNode.isLeaf == True:
            #print 'reached the inside of the if'
            #Bug here: What is getLabel() for  leaf? Is it ever even initialized? How do we know its correct?
            
            return curNode.getLabel()
        else:
            print 'Error, we didnt reach a leaf'
            
        #print 'finished prediction on this one example, exiting the pedict function'
        
    def predictAll(self, X):
        N,D = X.shape
        #print "the number of examples in X is"
        #print N
        Y   = zeros((N,), dtype=int)
        for n in range(N):
            #print n
            Y[n] = self.predict(X[n])
            #print 'prediction stored in Yn is'
            #print Y[n]
        return Y
        
def getMyTreeForWine():
    return makeBalancedTree(20)

