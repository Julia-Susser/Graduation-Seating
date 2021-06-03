import pandas as pd
import numpy as np
import random
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import copy

class Evaluation: 
    def VerticleHappiness(self, people, chart):
        HappyPeople = []
        for person in people:
            row, col = self.getRowColFromPerson(person)

            choices = self.getChoices(person)

            if row!=self.numRows-1:
                p = chart[row+1][col]
                rank = self.getRank(p,choices)

                if rank in range(1,6) and person not in HappyPeople:
                    HappyPeople.append(person)


            if row!=0:
                p = chart[row-1][col]

                rank = self.getRank(p,choices)
 
                if rank in range(1,6) and person not in HappyPeople:
                    HappyPeople.append(person)
        return HappyPeople
    def DoubleSideHappiness(self, people, chart):
        HappyPeople = []
        for person in people:
            row, col = self.getRowColFromPerson(person)

            choices = self.getChoices(person)

            if row!=self.numRows-1 and row!=self.numRows-2:
                p = chart[row+2][col]
                rank = self.getRank(p,choices)

                if rank in range(1,6) and person not in HappyPeople:
                    HappyPeople.append(person)


            if row!=0 or row!=1:
                p = chart[row-2][col]

                rank = self.getRank(p,choices)
 
                if rank in range(1,6) and person not in HappyPeople:
                    HappyPeople.append(person)
        return HappyPeople
            
    
    
    def EvaluateHappiness(self,chart):
        Score = {}
        Done = []
        for Level in range(1,6):
            
            HappyPeople = []
            NoPreferences = []
            UnhappyPeople = []
            
            for row in range(self.numRows):
                for col in range(self.numCols):
                    person = chart[row][col]
                    if person in [-1,0]: continue
                
                    if self.NaPerson(person): 
                        NoPreferences.append(person)
                        continue

                    choices = self.getChoices(person)
                    
                    if col!=self.numCols-1:
                        p = chart[row][col+1]
                        
                        rank = self.getRank(p,choices)
                        if rank == Level and person not in HappyPeople and person not in Done: 
                            HappyPeople.append(person)
                        
                        
                    if col!=0:
                        p = chart[row][col-1]
                        rank = self.getRank(p,choices)
                        
                        if rank == Level and person not in HappyPeople and person not in Done:
                            HappyPeople.append(person)
                        
            Score[Level] = HappyPeople
            Done += HappyPeople
        
        
        numUnhappy = 94-len(NoPreferences)-len(Done)
        self.UnhappyPeople = [p for p in range(1,95) if p not in NoPreferences+Done]
        self.HaveVerticle = self.VerticleHappiness(UnhappyPeople, chart)
        self.numVerticleHappy = len(self.HaveVerticle)
        self.HaveDoubleSide = self.DoubleSideHappiness(UnhappyPeople, chart)
        self.numDoubleSideHappy = len(self.HaveDoubleSide)
        self.NoPreferences = len(NoPreferences)
        self.numRankOne = len(Score[1])
    
    def Print(self):
        print(self.Chart)
        print("Side by Side: First Choice:", self.numRankOne)
        print("# of people No Preferences", self.NoPreferences)
        print("Not in Top 5:", self.UnhappyPeople)
        print("Verticle Happy:", self.HaveVerticle)
        print("Double Side Happy:", self.HaveDoubleSide)
    

        
        

class FunctionTools:
    def getRowColFromPerson(self,person):
        for row in range(self.numRows):
                for col in range(self.numCols):
                    p = self.Chart[row][col]
                    if person==p:
                        return row,col
                    
    def idealPairs(self):
        self.firsts = []
        for pair in self.matches:
            if pair["rank1"] in [1,2,None] and pair["rank2"] in [1,2,None]:
                self.firsts.append(pair)
    
    def makeDict(self, person1, rank1, person2, rank2):
        return {"person1":person1,"rank1":rank1,"person2":person2, "rank2":rank2}
    
    def getChoices(self,p):
        return list(self.df.loc[p])
    
    def removePersonFromMatches(self,person):
        return [pair for pair in matches if not (pair["person1"]==person or pair["person2"]==person)]
    
    def removePersonFromPeople(self,p, people):
        return [person for person in people if person != p]
        
                
    def getMatches(self,person):
        return [pair for pair in self.matches if pair["person1"]==person or pair["person2"]==person]
    
    def getRank(self,person,choices):
        if person in choices: 
            rank = choices.index(person)+1
            if rank in [1,2]:
                return 1
            if rank in [3,4]:
                return 2
            if rank == 5:
                return 3
        return None
    
    def NaPerson(self,person):
        choices = self.getChoices(person)
        return np.isnan(choices).all()         
    
    def getPeople(self,pair):
        return [pair["person1"],pair["person2"]]
    
    def CompositeScore(self, pair):
        rank1 = pair["rank1"]
        rank2 = pair["rank2"]
        totalRank = rank2*2 if rank1 == None else (rank1*2 if rank2==None else rank1+rank2)
        return totalRank
    
    def findOtherPerson(self,pair, person):
        return pair["person1"] if person != pair["person1"] else pair["person2"]
    
    def findMatchPeople(self,matchesLeft,personLeft, matchesAbove, personAbove, matchesDoubleLeft, personDoubleLeft):
        peopleLeft = [self.findOtherPerson(pair, personLeft) for pair in matchesLeft]
        peopleAbove = [self.findOtherPerson(pair, personAbove) for pair in matchesAbove]
        peopleDoubleLeft = [self.findOtherPerson(pair, personDoubleLeft) for pair in matchesDoubleLeft]
        return list(set(peopleLeft).union(set(peopleAbove)).union(set(peopleDoubleLeft)))
    
    def samePeople(self,pair, p1,p2):
        np1, np2 = pair["person1"],pair["person2"]
        if (np1 == p1 and np2==p2) or (np2 == p1 and np1==p2):
            return True
        return False
    
    
    def getMatch(self,p1,p2):
        match = [pair for pair in self.matches if self.samePeople(pair,p1,p2)]
        if len(match)==0:return None
        return match[0]
    
    def scorePair(self, pair):
        rank1 = pair["rank1"]
        rank2 = pair["rank2"]
        totalRank = rank2*2 if rank1 == None else (rank1*2 if rank2==None else rank1+rank2)
        return totalRank
    
class CreateSeatingArrangement(Evaluation, FunctionTools):
    df = pd.read_csv("../input/info.csv").iloc[:,1:]
    df = df.set_index("Your name")
    def __init__(self):
        self.numCols = 21
        self.numRows = 5
        self.matches()
        self.idealPairs()
        self.MatchesForLonelyPeople()
        self.stringPairs = []
        
        
        
        self.initiateChart()
        self.createString(self.people)
        self.EvaluateHappiness(self.Chart)
        self.Print()
    def initiateChart(self):
        self.people = list(self.df.index)
        self.Chart = np.zeros((self.numRows,self.numCols))
        for row in range(1,3):
            for col in range(8,12):
                self.Chart[row-1][col-1] = -1
        self.Chart[0][12-1] = 42
        self.Chart[0][13-1] = 15
        self.Chart[0][14-1] = 33
        for person in [42,15,33]:
            self.people = self.removePersonFromPeople(person, self.people)
         
    def matches(self):
        matches = []
        for person,rankings in self.df.iterrows():
                for rankedPerson in rankings:
                    rank1 = self.getRank(rankedPerson, list(rankings))
                    if not pd.isnull(rankedPerson):
                        OtherPersonChoices = self.getChoices(rankedPerson)
                        if person in OtherPersonChoices:
                            if person > rankedPerson: continue #don't want duplicates if they overlap it will be added by first person
                            rank2 = OtherPersonChoices.index(person)
                            matches.append(self.makeDict(person,rank1+1,rankedPerson,rank2))
                        elif np.isnan(OtherPersonChoices).all():
                            matches.append(self.makeDict(person,rank1+1,rankedPerson,None))
        self.matches = matches
    
    def NoMatches(self):
        haveMatch = reduce(lambda a,b: a+b, [self.getPeople(pair) for pair in self.matches])
        return [p for p in range(1,95) if p not in haveMatch and not self.NaPerson(p)]
    
    def MatchesForLonelyPeople(self):
        needMatch = self.NoMatches()
        for person in needMatch:
            rankings = self.getChoices(person)
            for rankedPerson in rankings:
                rank1 = self.getRank(rankedPerson, list(rankings))
                if not pd.isnull(rankedPerson):
                    self.matches.append(self.makeDict(person,rank1+1,rankedPerson,rank1+3))
                    
    def CompositeScore(self,person,personLeft, personAbove, personDoubleLeft):
        if len(self.getMatches(person))==1: return -100
        matchLeft = self.getMatch(person,personLeft)
        matchAbove = self.getMatch(person,personAbove)
        matchDoubleLeft = self.getMatch(person,personDoubleLeft)
        totalRankDoubleLeft = 0 if matchDoubleLeft == None else self.scorePair(matchDoubleLeft)
        totalRankLeft = 0 if matchLeft == None else self.scorePair(matchLeft)
        totalRankAbove = 0 if matchAbove == None else self.scorePair(matchAbove)
        
        if matchAbove == None:
            if matchLeft == None:#just Double left
                return 100+ totalRankDoubleLeft
            if matchDoubleLeft == None:#just left
                return totalRankLeft
            else:#left and double left
                return totalRankLeft-.5*(12-totalRankDoubleLeft)
            
        if matchLeft == None:
            if matchDoubleLeft == None:#just above
                return 200+totalRankAbove
            else:#above and double left
                return totalRankDoubleLeft+3*totalRankAbove
        if matchDoubleLeft == None:#above and left
            return totalRankLeft-.25*(12-totalRankAbove)
        
        return totalRankLeft-.5*(12-totalRankDoubleLeft)-.25*(12-totalRankAbove)
        
    def lowestCompositeScore(self, row,col, people):
        
        personLeft = 0 if col==0 else self.Chart[row][col-1]
        personAbove = 0 if row==0 else self.Chart[row-1][col]
        personDoubleLeft = 0 if row==0 or row==1 else self.Chart[row-2][col]
        if (personLeft==0 and personAbove==0): return None
        matchesLeft = self.getMatches(personLeft)
        matchesAbove = self.getMatches(personAbove)
        matchesDoubleLeft = self.getMatches(personDoubleLeft)
        
        peopleOptions = self.findMatchPeople(matchesLeft,personLeft,matchesAbove, personAbove, matchesDoubleLeft, personDoubleLeft)
        peopleOptions = [person for person in peopleOptions if person in people]
        if len(peopleOptions)==0: return None
        scores = [self.CompositeScore(person, personLeft, personAbove, personDoubleLeft) for person in peopleOptions]
        minScore = min(scores)
        bestPerson = peopleOptions[scores.index(minScore)]
        return bestPerson
        
    
    
    def newPosition(self,row,col):
        if col==self.numCols-1: 
            col =0
            row += 1
        else: col+=1
        return row,col
    def adjustedNewPosition(self,row,col):
        row,col = self.newPosition(row,col)
        if row>=self.numRows: return row,col
        while self.Chart[row][col] in [15,33,42,-1] and row<self.numRows:
            row, col = self.newPosition(row,col)
        return row,col
            
            
    def setLeftToZero(self,row,col):
        row,col = self.newPosition(row,col)
        while row!=self.numRows:
            self.Chart[row][col]=0
            row,col = self.newPosition(row,col)
            
    
    def createString(self, people, row=0, col=0):

        if row == self.numRows: return
        newPerson = self.lowestCompositeScore(row, col, people)
        nrow,ncol = self.adjustedNewPosition(row,col)
        
        if newPerson==None:
            if len(people)==0:
                self.setLeftToZero(row,col)
                return
            
            elif col==self.numCols-1:
                
                self.Chart[row][col] = 0
                self.createString(people,nrow, ncol)
                
            else: 
                
                #for x in people:
                #min(len(people),2)
                for newPerson in random.sample(people,1):
                    self.Chart[row][col] = newPerson
                    newpeople = self.removePersonFromPeople(newPerson, people)
                    self.createString(newpeople,nrow, ncol)
                

        else:  
        
            self.Chart[row][col] = newPerson
            newpeople = self.removePersonFromPeople(newPerson, people)

            self.createString(newpeople, nrow, ncol)
    
   
    
    
    
    
