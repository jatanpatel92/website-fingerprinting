#VNGPlusPlusClassifier

# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import sys
import config
import time
import os
import random
import getopt
import string
import itertools

# custom
from Datastore import Datastore
from Webpage import Webpage

# countermeasures
from PadToMTU import PadToMTU
from PadRFCFixed import PadRFCFixed
from PadRFCRand import PadRFCRand
from PadRand import PadRand
from PadRoundExponential import PadRoundExponential
from PadRoundLinear import PadRoundLinear
from MiceElephants import MiceElephants
from DirectTargetSampling import DirectTargetSampling
from Folklore import Folklore
from WrightStyleMorphing import WrightStyleMorphing

# classifiers
from LiberatoreClassifier import LiberatoreClassifier
from WrightClassifier import WrightClassifier
from BandwidthClassifier import BandwidthClassifier
from HerrmannClassifier import HerrmannClassifier
from TimeClassifier import TimeClassifier
from PanchenkoClassifier import PanchenkoClassifier
from VNGPlusPlusClassifier import VNGPlusPlusClassifier
from VNGClassifier import VNGClassifier
from JaccardClassifier import JaccardClassifier
from ESORICSClassifier import ESORICSClassifier

def run():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:T:N:k:n:r:h")
    except getopt.GetoptError, err:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)

    char_set = string.ascii_lowercase + string.digits
    runID = ''.join(random.sample(char_set,8))

    for o, a in opts:
        if o in ("-k"):
            config.BUCKET_SIZE = int(a)
        elif o in ("-N"):
            config.TOP_N = int(a)
        elif o in ("-t"):
            config.NUM_TRAINING_TRACES = int(a)
        elif o in ("-T"):
            config.NUM_TESTING_TRACES = int(a)
        elif o in ("-n"):
            config.NUM_TRIALS = int(a)
        elif o in ("-r"):
            runID = str(a)
        else:
            sys.exit(2)
    outputFilenameArray = ['model3',
                           'k'+str(config.BUCKET_SIZE),
                           'N'+str(config.TOP_N),
                           't'+str(config.NUM_TRAINING_TRACES),
                           'T'+str(config.NUM_TESTING_TRACES),
                          ]
    outputFilename = os.path.join(config.OUTPUT_DIR,'.'.join(outputFilenameArray))

    if not os.path.exists(config.CACHE_DIR):
        os.mkdir(config.CACHE_DIR)

    if not os.path.exists(outputFilename+'.output'):
        banner = ['accuracy','overhead','timeElapsedTotal','timeElapsedClassifier']
        f = open( outputFilename+'.output', 'w' )
        f.write(','.join(banner))
        f.close()
    if not os.path.exists(outputFilename+'.debug'):
        f = open( outputFilename+'.debug', 'w' )
        f.close()
    #JaccardClassifier - 2
    #VNGPlusPlusClassifier - 3
    #startIndex = config.NUM_TRAINING_TRACES
    #endIndex   = len(config.DATA_SET)-config.NUM_TESTING_TRACES
    print "VNGPlusPlus Classifier"
    print "---------------------"
    startIndex = 0
    endIndex   = 20
    retrainTime = []
    while (endIndex<len(config.DATA_SET)):
        for i in range(config.NUM_TRIALS):
            startStart = time.time()
            # k-privacy set selection
            webpageIds = range(0, config.TOP_N - 1)
            random.shuffle( webpageIds )
            webpageIds = webpageIds[0:config.BUCKET_SIZE]
            print "Sets of size k:"
            print webpageIds
            print "startIndex:%d"%startIndex
            print "endIndex:%d"%endIndex
            seed = random.randint( startIndex, endIndex )
            print "seed:%d"%seed
            preCountermeasureOverhead = 0
            postCountermeasureOverhead = 0
    
            #classifier     = intToClassifier(config.CLASSIFIER)
            classifier = VNGPlusPlusClassifier
            countermeasure = 0 
            #intToCountermeasure(config.COUNTERMEASURE)
            
            trainingSet = []
            testingSet  = []
    
            targetWebpage = None
    
            for webpageId in webpageIds:
                webpageTrain = Datastore.getWebpagesLL( [webpageId], seed-config.NUM_TRAINING_TRACES, seed )
                webpageTest  = Datastore.getWebpagesLL( [webpageId], seed, seed+config.NUM_TESTING_TRACES )
                webpageTrain = webpageTrain[0]
                webpageTest = webpageTest[0]
                if targetWebpage == None:
                    targetWebpage = webpageTrain
    
                preCountermeasureOverhead  += webpageTrain.getBandwidth()
                preCountermeasureOverhead  += webpageTest.getBandwidth()
    
                metadata = None
                i = 0
                for w in [webpageTrain, webpageTest]:
                    for trace in w.getTraces():
                        traceWithCountermeasure = trace
    
                        postCountermeasureOverhead += traceWithCountermeasure.getBandwidth()
                        instance = classifier.traceToInstance( traceWithCountermeasure )
    
                        if instance:
                            if i==0:
                                trainingSet.append( instance )
                            elif i==1:
                                testingSet.append( instance )
                    i+=1
    
            ###################
    
            startClass = time.time()
            [accuracy,debugInfo] = classifier.classify( runID, trainingSet, testingSet )
            acc = accuracy
            print "Accuracy:"
            print acc
            endIndex+=1
            end = time.time()
    
            overhead = str(postCountermeasureOverhead)+'/'+str(preCountermeasureOverhead)
    
            output = [accuracy,overhead]
    
            output.append( '%.2f' % (end-startStart) )
            output.append( '%.2f' % (end-startClass) )
    
            summary = ', '.join(itertools.imap(str, output))
    
            f = open( outputFilename+'.output', 'a' )
            f.write( "\n"+summary )
            f.close()
    
            f = open( outputFilename+'.debug', 'a' )
            for entry in debugInfo:
                f.write( entry[0]+','+entry[1]+"\n" )
            f.close()
        if (acc<=85):
            print "*****Shift The Training Window*****"
            retrainTime.append(endIndex-1)
            print "Re-training Days List:"
            print retrainTime
            startIndex = endIndex
            endIndex = startIndex+20
    print "Average Re-training Time Period for Model3:"
    sum = retrainTime[0];
    for i in range(1, len(retrainTime)):
        sum+=retrainTime[i]-retrainTime[i-1]
    avg = sum/len(retrainTime)
    print avg

if __name__ == '__main__':
    run()
