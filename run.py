# -*- coding:utf-8 -*-

import sys
import dtree


#==fc==
def get_training_file():
    training_filename = './train.dat'
    try:
        fTrainIn = open(training_filename, "r")
    except IOError:
        print("Error: Could not find the training file specified or unable "
              "to open it" % training_filename)
        sys.exit(0)
    return fTrainIn

#==fc==
def get_test_file():
    test_filename = "./test.dat"
    try:
        fTestIn = open(test_filename, "r")
    except IOError:
        print("Error: Could not find the test file specified or unable to "
              "open it" % test_filename)
        sys.exit(0)
    return fTestIn


#==fc==
def prepare_attributes(attrList):
    attrList = attrList[:]
    attrs = []
    attrsDict = {}
    for i in range(0, len(attrList)-1, 2):
        attrs.append(attrList[i])
        # set the value of attribute name as key to its number
        attrsDict[attrList[i]] = attrList[i+1]
    return attrs, attrsDict

# ==fc==
def run_app(fTrainIn, fTestIn):
    #==fc== file lines into python list
    linesInTest = [line.strip() for line in fTestIn.readlines()]
    #==fc== collect attribute
    attributes = linesInTest[0].split(" ")
    #==fc== get rid of the header line
    linesInTest.reverse()
    linesInTest.pop()    # pop()弹出并返回最后一行
    linesInTest.reverse()

    attrList, attrDict = prepare_attributes(attributes)
    targetAttribute = attrList[-1]

    # prepare testdata
    testData = []
    for line in linesInTest:
        testData.append(dict(list(zip(attrList,
                                      [datum.strip()
                                       for datum in line.split("\t")]))))

    linesInTrain = [lineTrain.strip() for lineTrain in fTrainIn.readlines()]
    attributesTrain = linesInTrain[0].replace("\t", " ").split(" ")

    #==fc== do the same for training data
    #once we have the attributes remove it from lines
    linesInTrain.reverse()
    linesInTrain.pop()   # pops from end of list, hence the two reverses
    linesInTrain.reverse()

    attrListTrain, attrDictTrain = prepare_attributes(attributesTrain)
    targetAttrTrain = attrListTrain[-1]

    # prepare data
    trainData = []
    for lineTrain in linesInTrain:
        trainData.append(dict(list(zip(attrListTrain,
                                       [datum.strip()
                                        for datum in lineTrain.split("\t")]))))

    possible_dic = {}
    for trainRecord in trainData:
        for k in trainRecord.keys():
            if k in possible_dic:
                possible_dic[k].append(trainRecord[k])
            else:
                possible_dic[k]=[trainRecord[k]]

    for kk in possible_dic.keys():
        possible_dic[kk] = dtree.unique(possible_dic[kk])

    trainingTree = dtree.create_decision_tree(None, trainData, attrListTrain,targetAttrTrain, possible_dic,  dtree.gain)

    trainingClassify = dtree.classify(trainingTree, trainData)

    #testTree = dtree.create_decision_tree(testData, attrList, targetAttribute,dtree.gain)

    testClassify = dtree.classify(trainingTree, testData)

    # also returning the example Classify in both the files
    givenTestClassify = []
    for row in testData:
        givenTestClassify.append(row[targetAttribute])

    givenTrainClassify = []
    for row in trainData:
        givenTrainClassify.append(row[targetAttrTrain])

    return (trainingTree, trainingClassify, testClassify, givenTrainClassify,
            givenTestClassify)

#==fc== calculate matched/total
def accuracy(algoClassify, targetClassify):
    matching_count = 0.0
    for alg, target in zip(algoClassify, targetClassify):
        if alg == target:
            matching_count += 1.0
    # print len(algoClassify)
    # print len(targetClassify)
    return (matching_count / len(targetClassify)) * 100


#==fc== print tree
def print_tree(tree, str):
    if isinstance(tree, dict):
        # print("%s%s = " % (str, list(tree.keys())[0]))
        for item in list(list(tree.values())[0].keys()):
            print("  |  %s%s = %s" % (str, list(tree.keys())[0], item))
            print_tree(list(tree.values())[0][item], str + "   ")
            # the space in 'str + "   "' affect Backspace between Sub-layer
    else:
        print("   -->  %s : %s" % (str, tree))
        # --> stand for the targetAttribute

#==fc== program entry point
if __name__ == "__main__":
    fTrainIn = get_training_file()
    fTestIn = get_test_file()
    (trainingTree, trainingClassify, testClassify, givenTrainClassify,givenTestClassify) = run_app(fTrainIn, fTestIn)
    print_tree(trainingTree, "")
    print(" Accuracy of training set (%s instances) :  %s"
          % (len(givenTrainClassify),
             accuracy(trainingClassify, givenTrainClassify)))
    print(" Accuracy of test set (%s instances) :  %s"
          % (len(givenTestClassify),
             accuracy(testClassify, givenTestClassify)))