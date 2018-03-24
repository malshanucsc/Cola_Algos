import numpy

def NN(m1,m2,w1,w2,b):
    z = (m1*w1)+(m2*w2)+b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()
    
output=NN(3,1.5,w1,w2,b)

print(output)
