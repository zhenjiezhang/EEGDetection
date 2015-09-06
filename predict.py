import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2


datum=caffe_pb2.Datum()
testDB=leveldb.LevelDB('databases/test_subj1_leveldb')

net=caffe.Classifier('deploy.prototxt','./snapshots/_iter_5000.caffemodel',image_dims=(64,32))
caffe.set_mode_gpu

for k, v in testDB.RangeIter():
    datum.ParseFromString(v)
    data=caffe.io.datum_to_array(datum).reshape(1,64,32,1)
    label=datum.label
    # print data.shape
    # print [(k,v[0].data.shape) for k,v in net.params.items()]
    prediction=net.predict(data)

    print label, prediction.argmax()


