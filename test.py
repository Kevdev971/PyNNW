import pynnw as pnn

net = pnn.NNW([2,6,3,1])

x = pnn.array([[1,-1],[-1,1]])
y = pnn.array([[1],[0]])

print("--Test forward no train--")
print(net.forward(x))

print("--Test train--")
net.train(x, y, 100000, 0.02)

print("--Test forward train--")
print(net.forward(pnn.array([[1,-1],[-1,1],[0,0]])))