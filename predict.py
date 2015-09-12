import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
from   sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import roc_curve as roc


datum=caffe_pb2.Datum()
testDB=leveldb.LevelDB('databases/test_subj1_leveldb')

net=caffe.Classifier('deploy.prototxt','./snapshots/_iter_5000.caffemodel',image_dims=(64,32))
caffe.set_mode_gpu

labels=[]
pred=[]

predSize=12

for k, v in testDB.RangeIter():
    datum.ParseFromString(v)
    data=caffe.io.datum_to_array(datum).reshape(1,64,32,1)
    label=datum.label
    # print data.shape
    # print [(k,v[0].data.shape) for k,v in net.params.items()]
    prediction=net.predict(data)

    labelList=np.zeros(predSize)
    labelList[label]=1
    labels.append(labelList)
    pred.append(prediction)

    print label, prediction.argmax()
labels=np.array(labels)
pred=np.array(pred).reshape(-1,predSize)
# print pred.max()

aucs=[]

for i in xrange(12):
	aucs.append(auc(labels[:][i], pred[:][i],average="micro"))
print aucs
print np.mean(aucs)

for i in xrange(12):
	print "gesture: ", i, ":"
	f,t,threasholds=roc(labels[:][i], pred[:][i])
	print f
	print t
	print threasholds





