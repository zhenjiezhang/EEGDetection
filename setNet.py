import caffe
from caffe import layers as L
from caffe import params as P

def setLayers(leveldb, batch_size, type):
    n = caffe.NetSpec()
    if type!="deploy":
        n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LEVELDB, source=leveldb,
                             ntop=2)
    else:
        input="data"
        dim1=1
        dim2=1
        dim3=64
        dim4=32
        n.data=L.Layer()

    n.conv2 = L.Convolution(n.data, kernel_h=6, kernel_w=1, num_output=8, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv2, kernel_h=3, kernel_w=1, stride_h=2, stride_w=1, pool=P.Pooling.MAX)

    n.drop2=L.Dropout(n.pool1,dropout_ratio=0.1)
    n.ip1=L.InnerProduct(n.drop2, num_output=196, weight_filler=dict(type='xavier'))

    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip4 = L.InnerProduct(n.relu1, num_output=12, weight_filler=dict(type='xavier'))

    if type!="deploy":
        n.accuracy=L.Accuracy(n.ip4,n.label)
        n.loss = L.SoftmaxWithLoss(n.ip4, n.label)
        return str(n.to_proto())
    else:
        n.prob=L.Softmax(n.ip4)
        deploy_str='input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"'+input+'"', dim1, dim2, dim3, dim4)
        return deploy_str+'\n'+'layer {'+'layer {'.join(str(n.to_proto()).split('layer {')[2:])


    
with open('trainNet.prototxt', 'w') as f:
    print 'wrting train'
    f.write(setLayers('databases/train_subj1_leveldb', 100, "train"))
    
with open('testNet.prototxt', 'w') as f:
    print 'wrting test'
    f.write(setLayers('databases/test_subj1_leveldb', 100, "test"))

with open('deploy.prototxt', 'w') as f:
    f.write(str(setLayers('', 0, "deploy")))

