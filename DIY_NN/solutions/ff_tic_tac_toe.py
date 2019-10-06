A = Matrix([1,1,1,0,0,0,0,0,0], 
           [0,0,0,1,1,1,0,0,0], 
           [0,0,0,0,0,0,1,1,1], 
           [1,0,0,1,0,0,1,0,0], 
           [0,1,0,0,1,0,0,1,0], 
           [0,0,1,0,0,1,0,0,1], 
           [1,0,0,0,1,0,0,0,1], 
           [0,0,1,0,1,0,1,0,0])#(8*9)

b = Vector(-2,-2,-2,-2,-2,-2,-2,-2)#(1*8)
C = Matrix([-1,-1,-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1,1,1])#(2*8)
d = Vector(1, 0) 
for i in range(10):
    board = np.random.randint(0,2, size=(3,3))
    board
    print( "\n".join("".join("_X"[k] for k in  board[j]) for j in range(3)) )
    x = Vector(board.ravel()) #(8*1)
    board.ravel()
    z = A@x+b
    q = softmax(C@relu(A@x+b)+d)
    print("q={}\n".format(q.argmax()))




