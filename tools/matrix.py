import numpy as np
def truncate_matrix(matrix,index,dim):
    shape=list(matrix.shape)
    shape[index]=dim
    newMatrix= matrix[0:shape[0],0:shape[1]]
    return newMatrix