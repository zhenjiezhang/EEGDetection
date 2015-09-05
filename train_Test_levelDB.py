import leveldb
import pandas as pd
import caffe
import numpy as np
import os
from caffe.proto import caffe_pb2

os.system('rm -rf train_leveldb_data')
os.system('rm -rf train_leveldb_label')
os.system('rm -rf test_leveldb_data')
os.system('rm -rf test_leveldb_label')




try:
    trainDataDB=leveldb.LevelDB('train_leveldb_data')
    trainLabelDB= leveldb.LevelDB('train_leveldb_label')
    testDataDB=leveldb.LevelDB('test_leveldb_data')
    testLabelDB=leveldb.LevelDB('test_leveldb_label')


    trainingFiles=os.listdir('train')

    num=0
    for f in trainingFiles:
        print f

        if 'data' in f:
            num+=1
            print 'file: ', num
            series=int(f.split('_')[1][-1])
            db_Data=testDataDB if series==8 else trainDataDB
            db_label=testLabelDB if series==8 else trainLabelDB


            batch_data=leveldb.WriteBatch()
            batch_label=leveldb.WriteBatch()
            batchSize=30000
            datum_data=caffe_pb2.Datum()
            datum_label=caffe_pb2.Datum()

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


            #set status to the latest action taken, reset status to 0 after one round of actions
            status=0
            for i in xrange(n):
                input=list(np.concatenate(([status],data[i]),axis=1))

                updateStatus=sum([k*labels[i][k-1] for k in range(1,7)])
                if updateStatus>status:
                    status=updateStatus
                elif status==7 and updateStatus==0:
                    status=updateStatus

                #convert data and label to datum structures
                datum_data=caffe.io.array_to_datum(np.array([[input]]))
                datum_label=caffe.io.array_to_datum(np.array([[labels[i]]]))

                #store data and label points to batches
                keystr= '{:0>2d}'.format(personID)+'{:0>2d}'.format(series)+'{:0>6d}'.format(i)
                batch_data.Put(keystr,datum_data.SerializeToString())
                batch_label.Put(keystr,datum_label.SerializeToString())

                #write batches into database
                if (i+1)%batchSize==0:
                    db_Data.Write(batch_data,sync=True)
                    db_label.Write(batch_label,sync=True)
                    batch_data=leveldb.WriteBatch()
                    batch_label=leveldb.WriteBatch()

                    print f, ': ', i+1

            if n%batchSize!=0:
                db_Data.Write(batch_data,sync=True)
                db_label.Write(batch_label,sync=True)

                print f, ' last : ', n




            # print 'ID=', personID, 'data', f.split('data.csv')[0], ':    ', data[0], 'events', labels[0]
finally:
    del trainDataDB
    del trainLabelDB
    del testDataDB
    del testLabelDB
