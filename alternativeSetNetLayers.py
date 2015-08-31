import caffe
from caffe import layers as L
from caffe import params as P

def setLayers(leveldb, batch_size):
    n = caffe.NetSpec()
    n.data = L.Data(batch_size=batch_size, backend=P.Data.LEVELDB, source=leveldb+'_data',
                             ntop=1)
    n.label=L.Data(batch_size=batch_size, backend=P.Data.LEVELDB, source=leveldb+'_label',
                             ntop=1)
    # n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    # n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    # n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.data, num_output=100, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=7, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()
    
with open('trainNet.prototxt', 'w') as f:
    f.write(str(setLayers('train_leveldb', 64)))
    
with open('testNet.prototxt', 'w') as f:
    f.write(str(setLayers('test_leveldb', 100)))