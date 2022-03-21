import caffe
from caffe import layers as cl, params as cp
def create_model():
    H=W=28
    net = caffe.NetSpec()
    net.data = cl.Input(shape=[dict(dim=[1, 3, H, W])])

    net.conv1 = cl.Convolution(net.data, name='conv2d', kernel_size=3, stride=1, num_output=6, pad=1)
    net.mp1 = cl.Pooling(net.conv1, kernel_size=2, stride=1, pool=cp.Pooling.MAX)
    net.conv2 = cl.Convolution(net.mp1, name='conv2d', kernel_size=3, stride=1, num_output=16, pad=1)
    net.mp2 = cl.Pooling(net.lr1, kernel_size=2, stride=1, pool=cp.Pooling.MAX)
    net.fl1 = cl.Flatten(net.mp2)
    net.den1 = cl.InnerProduct(net.fl1, num_output=120)
    net.relu1 = cl.ReLU(net.den1, in_place=True)
    net.den2 = cl.InnerProduct(net.relu1, num_output=84)
    net.relu2 = cl.ReLU(net.den2, in_place=True)
    net.den3 = cl.InnerProduct(net.relu2, num_output=10)
    net.soft = cl.Softmax(net.den3)

    with open('model_test.prototxt', 'w') as f: f.write(str(net.to_proto()))
    return net
