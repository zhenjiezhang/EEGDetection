import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
solver=caffe.SGDSolver('solver.prototext')

# solver.net.forward()
# print solver.net.blobs['label'].value


solver.solve()
# print solver.test_nets[0].forward()

# print [(k,v.data) for k, v in solver.net.blobs.items()]
# print 'parameters'
# print [(layer) for layer, p in solver.net.params.items()]
