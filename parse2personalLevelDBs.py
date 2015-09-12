import leveldb
import pandas as pd
import caffe
import os
import numpy as np
from caffe.proto import caffe_pb2
import sys


class subjectDataSet:
    def __init__(self, subjectNum):
        self.subjectNum=subjectNum
        self.subjectTrainingFiles=[['train/subj'+str(subjectNum)+'_series'+str(i)+'_data.csv' for i in xrange(1,9)]]+\
            [['train/subj'+str(subjectNum)+'_series'+str(i)+'_events.csv' for i in xrange(1,9)]]
        self.subjectPredictDataFiles=['test/subj'+str(subjectNum)+'_series'+str(i)+'_data.csv' for i in xrange(9,11)]
        self.setDataWindow(1024,16)

    def setDataWindow(self, winSize, stepSize):

        self.winSize=winSize
        self.stepSize=stepSize
        self.pointsInWindow=winSize//stepSize

    def readFile(self, series):
        print "reading subject "+str(self.subjectNum)+" series "+str(series)
        df=self.subjectTrainingFiles[0][series-1] if series <9 else self.subjectPredictDataFiles[series-9]

        data=pd.read_csv(df, index_col=0).values

        if series < 9:
            lf=self.subjectTrainingFiles[1][series-1]
            labels=pd.read_csv(lf,index_col=0).values
            return data, self.codeLabel(labels)
        else:
            return data

    def dataNormalization(self, data):
        mean=data.mean(axis=0)
        std=data.std(axis=0)
        return (data-mean)/std

    def makeWindowedData(self, data):
        stuffedData=np.vstack([[data[0] for i in xrange(self.winSize-1)],data])
        windowedData=[data[end-(self.pointsInWindow-1)*self.stepSize:end+1:self.stepSize]\
            for end in xrange(self.winSize-1, len(stuffedData))]
        return np.array(windowedData)



    def codeLabel(self, labels):
        codedLabels=[]
        lastStatus=0
        for label in labels:
            updateStatus=lastStatus
            if label[0]==1:
                updateStatus=1
            elif label[1]==1:
                updateStatus=2
                if label[2]==1:
                    updateStatus=3
                    if label[3]==1:
                        updateStatus=5
                elif label[3]==1:
                    updateStatus=4
            elif label[2]==1:
                updateStatus=6
                if label[3]==1:
                    updateStatus=7
            elif label[3]==1:
                updateStatus=8
            elif label[4]==1:
                updateStatus=9
                if label[5]==1:
                    updateStatus=10
            elif label[5]==1:
                updateStatus=11

            if lastStatus==11 and sum([a for a in label])==0:
                updateStatus=0

            lastStatus=updateStatus
            codedLabels.append(updateStatus)
        return np.array(codedLabels)

    def assembleData(self):
        print "assembling subjest"+str(self.subjectNum)
        self.trainData=None
        self.trainLabel=None
        self.testData=None
        self.testLabel=None
        self.predData=None


        trainSeries=range(1,8)
        testSeries=range(8,9)
        predSeries=range(9,11)

        for series in trainSeries:
            d,l=self.readFile(series)
            d=self.makeWindowedData(self.dataNormalization(d))
            if self.trainData is None:
                self.trainData=d
                self.trainLabel=l

            else:
                np.hstack([self.trainData,d])
                np.hstack([self.trainLabel,l])

        self.shuffledTrainList=np.arange(len(self.trainData))
        np.random.shuffle(self.shuffledTrainList)

        for series in testSeries:
            d,l=self.readFile(series)
            d=self.makeWindowedData(self.dataNormalization(d))
            if self.testData is None:
                self.testData=d
                self.testLabel=l
            else:
                np.hstack([self.testData,d])
                np.hstack([self.testLabel,l])

        for series in predSeries:
            d=self.readFile(series)
            d=self.makeWindowedData(self.dataNormalization(d))
            if self.predData is None:
                self.predData=d
            else:
                np.hstack([self.predData,d])



if __name__=="__main__":

    # sys.exit(0)



    subjects=xrange(1,13)
    for subject in subjects:



        dataSet=subjectDataSet(subject)
        dataSet.assembleData()


        trainDBFile='databases/train_subj'+str(dataSet.subjectNum)+'_leveldb'
        testDBFile='databases/test_subj'+str(dataSet.subjectNum)+'_leveldb'
        predDataDBFile='databases/predData_subj'+str(dataSet.subjectNum)+'_leveldb'

        for fileName in [trainDBFile,testDBFile,predDataDBFile]:
            os.system('rm -rf '+fileName)

        try:
            print "starting database for ", dataSet.subjectNum
            trainDB=leveldb.LevelDB(trainDBFile)
            testDB=leveldb.LevelDB(testDBFile)
            predDataDB=leveldb.LevelDB(predDataDBFile)

            for data, labels, db in zip([dataSet.trainData,dataSet.testData,dataSet.predData],\
                                        [dataSet.trainLabel,dataSet.testLabel,np.zeros(len(dataSet.predData))],
                                        [trainDB,testDB,predDataDB]):


            
                batch=leveldb.WriteBatch()
                batchSize=1000
                datum=caffe_pb2.Datum()  
       
                index=dataSet.shuffledTrainList if data is dataSet.trainData else xrange(data.shape[0])

                for i, count in zip(index, range(len(data))):

                    dataString=data[i][np.newaxis,:]

                    datum=caffe.io.array_to_datum(dataString,labels[i])

                    #store data and label points to batches
                    keystr= '{:0>10d}'.format(i)
                    batch.Put(keystr,datum.SerializeToString())

                    #write batches into database
                    if (count+1)%batchSize==0:
                        db.Write(batch,sync=True)
                        batch=leveldb.WriteBatch()

                        print 'database for subject: ',dataSet.subjectNum, db, ': ', count+1

                if len(datahub)%batchSize!=0:
                    db.Write(batch,sync=True)

                    print 'database for subject: ',dataSet.subjectNum, db, ' last : ', len(data)




        finally:
            del trainDB
            del testDB
            del predDataDB
