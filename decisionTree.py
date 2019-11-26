import sys 
import csv
import math

train_input = sys.argv[1]          
test_input = sys.argv[2]
depth = int(sys.argv[3])
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics = sys.argv[6]

def importData(train_input):
    features = {}
    labelList = []
    feat = []
    
    allData=open(train_input)
    reader = csv.reader(allData) 
        
    for row in reader:
        if (len(feat) == 0):
            for index in range (len(row)-1):
                feat.append([row[index]])
        else:
            for index in range (len(row)-1):
                feat[index].append(row[index])   
                    
            labelList.append(row[-1])
            
    for j in range(len(row)-1):
        temp = feat[j].pop(0)
        features[temp] = feat[j]
            
    tags = list(set(labelList))    
    return [labelList, features, tags]
      
def countNumber(labelList, tags):
    label0 = 0
    label1 = 0
    for index in labelList:
        if index == tags[0]:
            label0 += 1
        else:
            label1 += 1
    return label0, label1
    
def calEntropy(labelList, tags):
    label0, label1 = countNumber(labelList, tags)
    if(label0 == 0 or label1 == 0):
        return 0
   
    prob0 = 1.0*label0/len(labelList)
    prob1 = 1.0*label1/len(labelList)
    
    return -prob0*math.log(prob0,2) - prob1*math.log(prob1,2)
 
def calMutualInformation(feature, labelList):
    nLabels = []
    yLabels = []  
    tags = list(set(labelList)) 
         
    for index in range(len(feature)):
        if (feature[index] == ('n'or 'notA' or 'no')):
            nLabels.append(labelList[index])
        else:
            yLabels.append(labelList[index])
            
    prob1 = 1.0*(len(nLabels))/len(labelList)
    prob2 = 1.0*(len(yLabels))/len(labelList)
    
    return calEntropy(labelList, tags) - (prob1 *calEntropy(nLabels, tags))- (prob2 *calEntropy(yLabels, tags)), nLabels, yLabels
 
 
def splitFeatures(feature, features):
    nFeatures = {}
    yFeatures = {}
    len_list = len(features[feature])
    for index in features:
        if (index == feature):
            continue
        nFeatures[index] = []
        yFeatures[index] = []
        for j in range(len_list):
            if (features[feature][j] == ('n'or 'notA' or 'no')):
                nFeatures[index].append(features[index][j])
            else:
                yFeatures[index].append(features[index][j])

    return nFeatures, yFeatures


class Node(object):
    def __init__(self, tag, left = None, right = None, feature = None):
        self.tag = tag
        self.left = left
        self.right = right
        self.feature = feature
    
    def Leaf(self):
        if (self.left == None and self.right == None):
            return True
        else:
            return False
            
def DTtrain(labelList, features, tags, curdepth, maxdepth):
    label0num, label1num = countNumber(labelList, tags)

    if (label0num > label1num):
        predict = tags[0]
        predictnum = label0num
    else:
        predict = tags[1]
        predictnum = label1num

    # base case: no need to split further
    if (predictnum == len(labelList)):
        return Node(predict)

    # base case: cannot split further
    elif (len(features) == 0):
        return Node(predict)

    # do not split over max_depth
    elif (curdepth > maxdepth):
        return Node(predict)

    else:
        nlabels = []
        ylabels = []
        score = -1
        feature = []
        for i in features:
            # the accuracy we would get if we only queried on i
            currentscore, currentnlabels, currentylabels = calMutualInformation(features[i], labelList)

            if (currentscore >= score):
                score = currentscore
                nlabels = currentnlabels
                ylabels = currentylabels
                feature = i

        curdepth += 1
        nfeatures, yfeatures = splitFeatures(feature, features)


        left = DTtrain(nlabels, nfeatures, tags, curdepth, maxdepth)
        right = DTtrain(ylabels, yfeatures, tags, curdepth, maxdepth)

        return Node(tags[0], left, right, feature)

        
def DTtest(node, test_input, output):
    allData=open(train_input)
    reader = csv.reader(allData)
    
    count = 0
    total = 0
    data = []
    feat = []
    
    for row in reader:
        if (total == 0):
            for index in range(len(row)- 1):
                feat.append(row[index])
        else :
            dict = {}
            for index in range(len(row)- 1):
                dict[feat[index]] = row[index]
            
            label = test(node, dict)
            data.append(label + '\n')
            if (label != row[-1]):
                count = count + 1
        total = total +1
    str = "".join(data)
    with open(output,'w') as f:
        f.writelines(str)
    f.close()
    
    return 1.0*count/(total-1)
    
def test(node, dict):
    if (node.Leaf()):
        return node.tag
    else:
        if (dict[node.feature] == ('n'or 'notA' or 'no')):
            return test(node.left, dict)
        else:
            return test(node.right, dict)
            
def printTree(node, depth):
    if (node.Leaf()):
        print(' '*depth, node.tag)
    else:
        print(' '*depth, node.tag, ': ',node.feature)
        printTree(node.left, depth + 1)
        printTree(node.right, depth + 1)
        0

def trainandtest(train_input, test_input, depth, train_out, test_out, metrics):
    data = importData(train_input)
    train_labelList = data[0]
    train_features = data[1]
    train_tags = data[2]

    node = DTtrain(train_labelList, train_features, train_tags, 0, depth)
    train_error = DTtest(node, train_input, train_out)   
    test_error = DTtest(node, test_input, test_out)   
    
    str = 'error(train):{}\nerror(test):{}'.format(train_error, test_error)
    with open(metrics,'w') as f:
        f.writelines(str)
    f.close()
    
## MAIN FUNCTION
if __name__ == '__main__':
    trainandtest(train_input, test_input, depth, train_out, test_out, metrics)