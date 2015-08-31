import leveldb
import pandas as pd
import caffe
import numpy as np
import os
from caffe.proto import caffe_pb2

os.system('rm -rf databases/train_leveldb')
os.system('rm -rf databases/test_leveldb')




try:
    trainDB=leveldb.LevelDB('databases/train_leveldb')
    testDB=leveldb.LevelDB('databases/test_leveldb')


    trainingFiles=os.listdir('train')

    num=0
    for f in trainingFiles:
        print f

        if 'data' in f:
            num+=1
            print 'file: ', num
            series=int(f.split('_')[1][-1])
            db=testDB if series==8 else trainDB

            batch=leveldb.WriteBatch()
            batchSize=30000
            datum=caffe_pb2.Datum()

            personID=int(f.split('_')[0][4:])
            data=pd.read_csv('train/'+f)

            #convert the data time id to int
            data['id']=data['id'].apply(lambda s: int(s.split('_')[2]))
            data=data.values

            n=len(data)
            #add person ID as a feature
            data=np.concatenate(([[personID] for i in xrange(n)],data),axis=1).astype(int)
            #remove id in labels
            labels=pd.read_csv('train/'+f.replace('data','events')).values[:,1:].astype(int)

            # normalize data
            data=data-data.mean(axis=0)
            data=data/data.std(axis=0)


            #set lastStatus to the latest action taken, reset lastStatus to 0 after one round of actions
            lastStatus=0
            for i in xrange(n):
                input=list(np.concatenate(([lastStatus],data[i]),axis=1))

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


                #convert data and label to datum structures
                datum=caffe.io.array_to_datum(np.array([[input]]),updateStatus)

                #store data and label points to batches
                keystr= '{:0>2d}'.format(personID)+'{:0>2d}'.format(series)+'{:0>6d}'.format(i)
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
