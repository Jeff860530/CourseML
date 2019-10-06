def softmax(x):
    t = np.exp(x)
    return t/t.sum()

import numpy as np

%run magic.ipynb


def relu(x):
    return np.maximum(x, 0)

A = Matrix([0,0,0,0], 
           [1,-1,1,-1], 
           [-1,1,-1,1],
           [-10,10,-10,10],
           [10,-10,10,-10],
          )
b = Vector(0.1,0,0,-12,-12)

C = Matrix([1,0,0,0,0], 
           [0,1,0,1,0], 
           [0,0,1,0,1],
          )
d = Vector(0,0,0) 
count = 0
unmber = 16
for i in range(unmber):
    x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
    print(x)
    q = softmax(C@relu(A@x+b)+d)
    print(q)
    print("i={}, i%3={}, q={}".format(i, i%3, q.argmax()))
    if( q.argmax()== (i%3) ):
      count+=1
print("Accuracy:{}%".format(count*100/unmber))