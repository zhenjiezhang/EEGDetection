import caffe
from caffe import layers as L
from caffe import params as P

def setLayers(leveldb, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LEVELDB, source=leveldb,
                             ntop=2)

    # n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    # n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    # n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    # n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # n.drop1=L.Dropout(n.data,dropout_ratio=0.5)
    # n.conv1 = L.Convolution(n.drop1, kernel_size=1, num_output=4, weight_filler=dict(type='xavier'))

    n.conv2 = L.Convolution(n.data, kernel_h=5, kernel_w=1, num_output=8, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv2, kernel_h=2, kernel_w=1, stride_h=2, stride_w=1, pool=P.Pooling.MAX)

    n.drop2=L.Dropout(n.pool1,dropout_ratio=0.3)
    n.ip1 = L.InnerProduct(n.drop2, num_output=64, weight_filler=dict(type='xavier'))
    # n.drop3=L.Dropout(n.ip1,dropout_ratio=0.5)
    # n.ip2 = L.InnerProduct(n.drop3, num_output=196, weight_filler=dict(type='xavier'))
    # n.drop4=L.Dropout(n.ip2,dropout_ratio=0.5)
    # n.ip3 = L.InnerProduct(n.ip1, num_output=12, weight_filler=dict(type='xavier'))    

    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip4 = L.InnerProduct(n.relu1, num_output=12, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip4, n.label)
    return n.to_proto()
    
with open('trainNet.prototxt', 'w') as f:
    f.write(str(setLayers('databases/train_subj1_leveldb', 100)))
    
with open('testNet.prototxt', 'w') as f:
    f.write(str(setLayers('databases/test_subj1_leveldb', 100)))