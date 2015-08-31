import os
import caffe

os.chdir('/home/ubuntu/caffe/')
caffe.set_device(0)
caffe.set_mode_gpu()
solver=caffe.SGDSolver('/home/ubuntu/caffe/examples/mnist/lenet_solver.prototxt')

solver.net.forward()
solver.test_nets[0].forward()

print solver.net.blobs['label'].data[:3]
print solver.net.blobs['ip2'].data[:3]
print solver.net.blobs['loss'].data