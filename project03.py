import numpy as np
import copy
import random

result =[]
#The function to get the combinations of index to produce the input test vectors
def getCombinations(pattern,path,index,misclassifiedLength):
    if misclassifiedLength == 0:
        temp =copy.deepcopy(path)
        result.append(temp)
        return
    for i in range(index,6):
        path.append(i)
        getCombinations(pattern,path,i+1,misclassifiedLength-1)
        path.pop()

#The function to get the 64 input test vectors
flippedPatterns =[]
def flipping(combinationIndex,pattern):
    for i in range(len(combinationIndex)):
        temp = copy.deepcopy(pattern)   #reference by value
        for j in range(len(combinationIndex[i])):
            num = -1*temp.item(combinationIndex[i][j])
            temp.itemset(combinationIndex[i][j],num)
        flippedPatterns.append(temp)

#The function to reverse a given pattern.
def reversePattern(pattern):
    temp = copy.deepcopy(pattern)   #reference by value
    for j in range(6):
        num = -1*temp.item(j)
        temp.itemset(j,num)
    return temp

#the neural network training
class neural:
    def training(self,trainData):
        model = {}
        weight = np.matrix('0,0,0,0,0,0;0,0,0,0,0,0;0,0,0,0,0,0;0,0,0,0,0,0;0,0,0,0,0,0;0,0,0,0,0,0')   #initial weight matrix
        for i in range(len(trainData)):
            fresh = trainData[i].transpose()
            weight = weight + (fresh*trainData[i])
        for j in range(6):      #setting the diagonals of weight to be 0
            temp = weight[j]
            num = 0*temp.item(j)
            temp.itemset(j,num)
        model['weight'] = weight
        return model

#calculating the output from the input test vector
    def testing(self, y, model):
        weight = model['weight']
        x = copy.deepcopy(y)  # reference by value
        ytemp = copy.deepcopy(y)
        flag = 0
        count = 0
        neurons = [0,0,0,0,0,0]     #array is created to check if all the activations are changing or not
        while (flag ==0):  # to make sure the loop goes on until all the neurons are checked
            yOld = copy.deepcopy(ytemp)
            count = count + 1
            num = random.randint(0, 5)  # choosing a random neuron
            sum = 0
            yVal = x[:, num] + ytemp * weight[:, num]  # updating y
            if (yVal > 0).all():  # activation
                yVal = 1
            elif (yVal < 0).all():
                yVal = -1
            else:
                yVal = yOld[:, num]
            ytemp[:, num] = yVal
            if (ytemp == yOld).all():
                neurons[num] = 1
            else:
                neurons[num] = 0
            for k in range(6):
                sum = sum+neurons[k]
            if(sum ==6):
                flag =1
            else:
                flag = 0
        return ytemp

def main():
    train1 = np.matrix('1,1,-1,-1,-1,1')        #training data set
    train2 = np.matrix('1,-1,-1,1,-1,-1')
    train3 = np.matrix('-1,-1,1,1,1,-1')
    train4 = np.matrix('-1,1,1,-1,1,1')
    trainData = [train1,train2,train3,train4]
    network = neural()
    model = network.training(trainData)         #produce the weights
    print("Checking for equilibrium states of the given input vectors : ")
    print("For training vector 1 :")
    output1 = network.testing(train1,model)
    print("Train Vector1 : ",train1)
    print("Output from model : ",output1)
    if(output1 == train1).all():
        print("Stored vector 1 is in equilibrium State")
    else:
        print("Not in Equilibrium")
    print("For training vector 2 :")
    output2 =network.testing(train2, model)
    print("Train Vector2 : ", train2)
    print("Output from model : ", output2)
    if(output2 == train2).all():
        print("Stored vector 2 is in equilibrium State")
    else:
        print("Not in Equilibrium")
    print("For training vector 3 :")
    output3 =network.testing(train3, model)
    print("Train Vector3 : ", train3)
    print("Output from model : ", output3)
    if(output3 == train3).all():
        print("Stored vector 3 is in equilibrium State")
    else:
        print("Not in Equilibrium")
    print("For training vector 4 :")
    output4 = network.testing(train4, model)
    print("Train Vector4 : ", train4)
    print("Output from model : ", output4)
    if(output4 == train4).all():
        print("Stored vector 4 is in equilibrium State")
    else:
        print("Not in Equilibrium")
    print("-------------------------------------------------------------------------------------------")

    pattern1 = np.matrix('1,1,1,1,1,1')         #producing all the input test vectors
    getCombinations(pattern1, [], 0, 0)
    getCombinations(pattern1, [], 0, 1)
    getCombinations(pattern1, [], 0, 2)
    getCombinations(pattern1, [], 0, 3)
    getCombinations(pattern1, [], 0, 4)
    getCombinations(pattern1, [], 0, 5)
    getCombinations(pattern1, [], 0, 6)
    flipping(result,pattern1)
    print(len(flippedPatterns))
    equilibriumPatterns = []
    print("The test input vectors used are total 64 :")
    for i in range(len(flippedPatterns)):       #printing all the test vectors
        print(flippedPatterns[i])

    for i in range(len(flippedPatterns)):
        out = network.testing(flippedPatterns[i],model)     #calculaating equilibrium state
        equilibriumPatterns.append(out)
    print("All equilibrium states from the 64 test vectors produced are :")
    for i in range(len(equilibriumPatterns)):           #printing equilibrium state
        print(equilibriumPatterns[i])
    print("Total equilibrium states : " ,len(equilibriumPatterns))

    unique = np.unique(np.array(equilibriumPatterns),axis=0)
    print("The unique equilibrium states are :")
    for pat in unique:
        print(pat)


    spurios = []
    for i in range(len(unique)):           #get spurios vectors
        flag = 0
        for j in range(len(trainData)):
            if(equilibriumPatterns[i]==trainData[j]).all():
                flag = 1
                break
        if(flag == 0):
            spurios.append(equilibriumPatterns[i])
    print("Total Spurios patterns are : ", len(spurios))

    reverseStates = []              #check for reverse of stored vectors
    for i in range(len(spurios)):
        temp = reversePattern(spurios[i])
        flag = 0
        for j in range(len(trainData)):
            if(temp == trainData[j]).all():
                flag = 1
                break
        if(flag == 1):
            reverseStates.append(temp)
    if(len(reverseStates)>0):
        print("The Spurios vectors which are reverse states of the stored vectors are:")
        for i in range(len(reverseStates)):
            print("Vector: " ,reverseStates[i] , " is the reverse of " , reversePattern(reverseStates[i]))
    else:
        print("None of the spurios vector is seen as the reverse of the stored vectors")

    print("---------------------------------------------------------------------------------------------")

    print("Finding the Basin of attraction for all the equilibrium states")
    c = 0
    for i in range(len(unique)):
        convergingPatterns = []
        temp = unique[i]
        for pat in flippedPatterns:
            out = network.testing(pat,model)
            if(out == temp).all():
                convergingPatterns.append(pat)
        print("The size of basin of attraction of " , temp, " is ",len(convergingPatterns),". Basin is = ")
        c = c+len(convergingPatterns)
        for conv in convergingPatterns:
            print(conv)

    print("----------------------------------------------------------------------------------------------")

    patternsNotAssociated = 0           #calculating the patterns that do not converge to the stored vectors
    for pat in flippedPatterns:
        out = network.testing(pat,model)
        flag = 0
        for j in range(len(trainData)):
            if(out == trainData[j]).all():
                flag = 1
                break
        if(flag == 0):
            patternsNotAssociated = patternsNotAssociated+1

    print(patternsNotAssociated,"patterns not associated")
    print("Chance is ", patternsNotAssociated/64)



if __name__ == '__main__':
    main()