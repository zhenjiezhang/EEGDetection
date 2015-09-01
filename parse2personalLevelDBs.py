import leveldb
import pandas as pd
import caffe
import os
import numpy as np
from caffe.proto import caffe_pb2
import sys

os.system('rm -rf databases/train_subj1_leveldb')
os.system('rm -rf databases/test_subj1_leveldb')




try:
    trainDB=leveldb.LevelDB('databases/train_subj1_leveldb')
    testDB=leveldb.LevelDB('databases/test_subj1_leveldb')

    subj1Files=['subj1_series'+str(i)+'_data.csv' for i in xrange(1,9)]
    # print subj1Files
    # sys.exit()

    # trainingFiles=os.listdir('train')
    subj1Train=[]

    subj1Test=[]





    for f in subj1Files:
        print f

        series=int(f.split('_')[1][-1])
        dataSet=subj1Test if series==8 else subj1Train

        # personID=int(f.split('_')[0][4:])
        data=pd.read_csv('train/'+f)

        #convert the data time id to int
        # data['id']=data['id'].apply(lambda s: int(s.split('_')[2]))
        data=data.values

        n=len(data)
        #add person ID as a feature
        # data=np.concatenate(([[personID] for i in xrange(n)],data),axis=1).astype(int)
        #remove id in labels
        labels=list(pd.read_csv('train/'+f.replace('data','events')).values[:,1:].astype(int))

        # normalize data
        # ids=data[:,0]
        # data=data[:,1:]
        data=data-data.mean(axis=0)
        data=data/data.std(axis=0)
        # data=np.concatenate(([[k]for k in ids],data),axis=1)

        lastStatus=0
        for i in xrange(n):
            data[i]+=[lastStatus]

            updateStatus=lastStatus
            if labels[i][0]==1:
                updateStatus=1
            elif labels[i][1]==1:
                updateStatus=2
                if labels[i][2]==1:
                    updateStatus=3
                    if labels[i][3]==1:
                        updateStatus=5
                elif labels[i][3]==1:
                    updateStatus=4
            elif labels[i][2]==1:
                updateStatus=6
                if labels[i][3]==1:
                    updateStatus=7
            elif labels[i][3]==1:
                updateStatus=8
            elif labels[i][4]==1:
                updateStatus=9
                if labels[i][5]==1:
                    updateStatus=10
            elif labels[i][5]==1:
                updateStatus=11

            if lastStatus==11 and sum([a for a in labels[i]])==0:
                updateStatus=0

            lastStatus=updateStatus
            labels[i] =updateStatus
        labels_data=np.concatenate(([[k] for k in labels], data),axis=1)
        dataSet+=[labels_data]

    subj1Train=np.concatenate(subj1Train,axis=0)
    subj1Test=np.concatenate(subj1Test,axis=0)

    

        # np.random.shuffle(labels_data)

        # data=labels_data[:,6:]
        # labels=labels_data[:,:6]


        #set lastStatus to the latest action taken, reset lastStatus to 0 after one round of actions
    
    
    #subsample:
    subsampleRate=16
    subsampleTrain=subj1Train[::16]
    subsampleTest=subj1Test[::16]

    #combine a time window of data as one record (window size counted in the number of data points in original traces).
    windowSize=1024
    subsampleWindowSize=windowSize/subsampleRate


    windowedTrain=np.array([subsampleTrain[i:i+subsampleWindowSize] for i in xrange(len(subsampleTrain)//subsampleWindowSize)])
    windowedTest=np.array([subsampleTest[i:i+subsampleWindowSize] for i in xrange(len(subsampleTest)//subsampleWindowSize)])

    batch=leveldb.WriteBatch()
    batchSize=1000
    datum=caffe_pb2.Datum()  


    print 'start'
    for dataSet, db in zip([windowedTrain, windowedTest],[trainDB,testDB]):
        np.random.shuffle(dataSet)
        print 'writing'

        for i in xrange(len(dataSet)):
            if i==1:
                print dataSet[i,:,1:].shape

            #convert data and label to datum structures
            datum=caffe.io.array_to_datum(np.array([dataSet[i,:,1:]]),int(dataSet[i,-1,0]))

            #store data and label points to batches
            keystr= '{:0>10d}'.format(i)
            batch.Put(keystr,datum.SerializeToString())

            #write batches into database
            if (i+1)%batchSize==0:
                db.Write(batch,sync=True)
                batch=leveldb.WriteBatch()

                print f, ': ', i+1

        if n%batchSize!=0:
            db.Write(batch,sync=True)

            print f, ' last : ', n




            # print 'ID=', personID, 'data', f.split('data.csv')[0], ':    ', data[0], 'events', labels[0]
finally:
    del trainDB
    del testDB
