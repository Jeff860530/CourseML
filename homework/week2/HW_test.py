#define function
def softmax(x):
    t = np.exp(x)
    return t/t.sum()
import numpy as np
%run magic.ipynb

def relu(x):
    return np.maximum(x, 0)


def Random_Matrix(row,column,start,finish):   
    #matrix = [np.array(np.random.randint(start,finish, size=column)) for i in range(row)]
    matrix = np.random.randint(start,finish, size=(column,row))
    return np.array(matrix)

def test_matrix_accuracy_4(times):
    best_count = 0
    best_matrix= np.array([])
    for i in range(times):
        W = Random_Matrix(4,4,-2,2)
        b = Vector(0,0,0,0)
        count = 0
        for i in range(16):
            x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
            r = W @ x + b
            #print("i=", i, "predict:", r.argmax(), "ground truth:", i%4)
            if( r.argmax()== (i%4) ):
              count+=1    
        if(count>best_count):
            best_count=count
            best_matrix = W     
    return best_count,best_matrix


def test_matrix_accuracy_3(times):
    best_count = 0
    best_matrix_A= np.array([])
    best_matrix_C= np.array([])
    B = Vector(0.1,0,0,-12,-12)
    D = Vector(0,0,0) 
    for i in range(times):
        A = Random_Matrix(5,4,-3,3)
        C = Random_Matrix(3,5,-3,3)
        count = 0
        for i in range(16):
            x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
            q = softmax(C@relu(A@x+B)+D)
            #print("i={}, i%3={}, q={}".format(i, i%3, q.argmax()))
            if( q.argmax()== (i%3) ):
              count+=1 
        if(count>best_count):
            best_count=count
            best_matrix_A = A
            best_matrix_C = C
    return best_count,best_matrix_A,best_matrix_C


times=100000
best_count,best_matrix=test_matrix_accuracy_4(times)
print("Best Accuracy:{}%".format(best_count*100/16))
print("Best Matric:")
print(best_matrix)


times=100000
best_count,best_matrix_A,best_matrix_C=test_matrix_accuracy_3(times)
print("Best Accuracy:{}%".format(best_count*100/16))
print("Best Matric A:")
print(best_matrix_A)
print("Best Matric C:")
print(best_matrix_C)


print("------------------------------------------")



import numpy as np

def Random_Matrix(row,column,start,finish):   
    matrix = np.random.randint(start,finish, size=(column,row))
    return np.array(matrix)
Matrix = []
Matrix.append([[1, 1, 1],[0, 0, 0],[0, 0, 0]])
Matrix.append([[0, 0, 0],[1, 1, 1],[0, 0, 0]])
Matrix.append([[0, 0, 0],[0, 0, 0],[1, 1, 1]])
Matrix.append([[1, 0, 0],[1, 0, 0],[1, 0, 0]])
Matrix.append([[0, 1, 0],[0, 1, 0],[0, 1, 0]])
Matrix.append([[0, 0, 1],[0, 0, 1],[0, 0, 1]])
Matrix.append([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
Matrix.append([[0, 0, 1],[0, 1, 0],[1, 0, 0]])
Matrix.append([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
Matrix_array = np.array(Matrix)

Matrix_array[8]
test_Matrix= Random_Matrix(3,3,0,2)

def find_Matrix_situation(test_Matrix,Matrix_array):
  Matrix_situation=np.array([1,0])
  sumofmatrix=0 
  for i in range(8):
      Matrix_multiplication=test_Matrix*Matrix_array[i]
      sumofmatrix = Matrix_multiplication.sum()
      if(sumofmatrix==3):
          Matrix_situation=np.array([0,1])
          return Matrix_situation,i,sumofmatrix
  return Matrix_situation,8,sumofmatrix   

def test(times):
  for j in range(times):
    test_Matrix= Random_Matrix(3,3,0,2)
    Matrix_situation,i,sumofmatrix=find_Matrix_situation(test_Matrix,Matrix_array)
  return Matrix_situation,test_Matrix

test(100)

def test_matrix_accuracy_OOXX(matrixtimes,ooxxtimes):
    best_count = 0
    best_matrix_A= np.array([])
    b = Vector(-2,-2,-2,-2,-2,-2,-2,-2)
    C = Matrix([-1,-1,-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1,1,1])
    d = Vector(1, 0) 
    for i in range(matrixtimes):
        A = Random_Matrix(8,9,0,2)
        count = 0
        for j in range(ooxxtimes):
            board = Random_Matrix(3,3,0,2)
            board
            #print( "\n".join("".join("_X"[k] for k in  board[j]) for j in range(3)) )
            x = Vector(board.ravel())
            board.ravel()
            z = A@x+b
            q = softmax(C@relu(A@x+b)+d)
            if( q.argmax()== Matrix_situation[1] ):
                count+=1 
            print(board)
            print("q.argmax():",q.argmax())
            print("Matrix_situation[1]:",Matrix_situation[1])
        print("Accuracy:{}%".format(count*100/ooxxtimes))
        if(count>best_count):
            best_count=count
            best_matrix_A = A
    return best_count,best_matrix_A

ooxxtimes=70
matrixtimes=10000
best_count,best_matrix_A=test_matrix_accuracy_OOXX(matrixtimes,ooxxtimes)
print("Best Accuracy:{}%".format(best_count*100/ooxxtimes))
print("Best_matrix_A",best_matrix_A)
#A= Random_Matrix(8,9,0,2)



A = Matrix([1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1],
           [0 ,1 ,1 ,1 ,0 ,1 ,0 ,1 ,0],
           [1 ,0 ,1 ,0 ,0 ,0 ,1 ,0 ,0],
           [1 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,0],
           [0 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,0],
           [1 ,0 ,0 ,1 ,1 ,0 ,1 ,0 ,0],
           [0 ,1 ,0 ,1 ,0 ,1 ,1 ,0 ,1],
           [1 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,0])

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



