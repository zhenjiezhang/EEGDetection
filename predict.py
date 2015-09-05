import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2


datum=caffe_pb2.Datum()
testDB=leveldb.LevelDB('databases/test_subj1_leveldb')

net=caffe.Classifier('testNet.prototxt','./snapshots/_iter_5000.caffemodel')
net.set_mode_gpu

for k, v in testDB.RangeIter():
    datum.ParseFromString(v)
    data=caffe.io.datum_to_array(datum)
    predictoin=net.predict(data)

    print data
    print prediction


