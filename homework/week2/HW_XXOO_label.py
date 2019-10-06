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



test_Matrix= Random_Matrix(3,3,0,2)
Matrix_situation,i,sumofmatrix=find_Matrix_situation(test_Matrix,Matrix_array)
print("test_Matrix:")
print(test_Matrix)
print("match_pattern:")
print(np.array(Matrix_array[i]))
print("Matrix_multiplication")
if (i<8):
    print(test_Matrix*Matrix_array[i])
else:
    print("[0]")
print("sumofmatrix",sumofmatrix)
print("Matrix_situation:",Matrix_situation)



