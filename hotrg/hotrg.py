import types 
import numpy as np 
from ncon import ncon 
import tensorly as tl
from hotrg_tools.matrix import truncate_matrix


class General_Node:
    # original_node = np.ndarray()
    # courrent_node = np.ndarray()
    transformation_log =  ""
    factor = 1

    def __init__(self,name,generator,*parameters ,verbose = 0):
        self.name = name
        self.verbose = verbose 
        self.parameters = parameters 
        self.generate = generator 
        self.original_node = self.generate(*self.parameters)
        self.courrent_node = self.generate(*self.parameters)
        self.transformation_log+="Node Generated\n"
        if self.verbose == 1:
            self.transformation_log.splitlines()[-1]
        pass

    def renew(self,*parameters):
        self.parameters = parameters
        self.original_node = self.generate(*self.parameters)
        self.courrent_node = self.generate(*self.parameters)
        self.factor = 1
        self.transformation_log = "Node genere"
        
    def self_contract(self,order_1,order_2):

        if type(order_1)==type(order_2) and  type(order_1) is list:

            if len(order_1)==len(order_2) and len(order_1) == len(self.courrent_node.shape):

                self.courrent_node = ncon([self.courrent_node,self.courrent_node],[order_1,order_2])
                self.transformation_log+=f"Contraction with ordering {order_1} {order_2},new {self.courrent_node.shape}\n"
                return self.courrent_node
            else:
                raise Exception("(Proto) Invalid lenght of order1/2.")

        else:
            raise Exception("(Proto) self_contract params must be lists")

    def reshape(self,dim_tuple):

        if type(dim_tuple) == tuple:
            self.courrent_node = self.courrent_node.reshape(dim_tuple)
            self.transformation_log+=f"Rehsaping with dimentions {dim_tuple},new {self.courrent_node.shape}\n"
            return self.courrent_node

        else:

            raise Exception("Argument must be tuple.")

    def truncate(self):
        pass

    def step(self,number_of_steps,dimension):
        pass

    def factorize(self):
        pass

    def trace(self):
        pass 

class Cross_Node(General_Node):
    """Expand general node to allow Xie Type Step and Truncation"""
    """For nodes with for legs of equal order"""
    def truncate(self,direction,dimension):

        """assumes initial direction xx'yy'"""
        if type(direction) is str:

            if direction == 'x':

                if self.courrent_node.shape[0]>dimension and self.courrent_node.shape[0] == self.courrent_node.shape[1]:
                    unfolded_buffer = np.matrix(tl.unfold(self.courrent_node, 0))
                    unfolded_buffer = unfolded_buffer @ unfolded_buffer.getH()
                    Ul, deltaL, _ = np.linalg.svd(unfolded_buffer)

                    del unfolded_buffer

                    unfolded_buffer = unfolded_buffer = np.matrix(tl.unfold(self.courrent_node, 1))
                    unfolded_buffer = unfolded_buffer @ unfolded_buffer.getH()
                    Ur, deltaR, _ = np.linalg.svd(unfolded_buffer)

                    del unfolded_buffer

                    eps1, eps2 = 0, 0

                    for i in range(dimension, len(deltaL)):
                        eps1 += deltaL[i]
                        eps2 += deltaR[i]
                    del deltaL, deltaR
                    if eps1 < eps2:
                        del Ur
                        Utr = np.array(truncate_matrix(Ul, 1, dimension))
                        del Ul
                    else:
                        del Ul
                        Utr = np.array(truncate_matrix(Ur, 1, dimension))
                        del Ur
                    self.courrent_node = ncon([Utr, Utr, self.courrent_node], [
                        [1, -1], [2, -2], [1, 2, -3, -4]])
                    return self.courrent_node

            elif direction == 'y':

                if self.courrent_node.shape[2]>dimension and self.courrent_node.shape[3] == self.courrent_node.shape[2]:

                    unfolded_buffer = np.matrix(tl.unfold(self.courrent_node, 2))
                    unfolded_buffer = unfolded_buffer @ unfolded_buffer.getH()
                    Ul, deltaL, _ = np.linalg.svd(unfolded_buffer)

                    del unfolded_buffer

                    unfolded_buffer = unfolded_buffer = np.matrix(tl.unfold(self.courrent_node, 3))
                    unfolded_buffer = unfolded_buffer @ unfolded_buffer.getH()
                    Ur, deltaR, _ = np.linalg.svd(unfolded_buffer)

                    del unfolded_buffer

                    eps1, eps2 = 0, 0

                    for i in range(dimension, len(deltaL)):
                        eps1 += deltaL[i]
                        eps2 += deltaR[i]
                    del deltaL, deltaR
                    if eps1 < eps2:
                        del Ur
                        Utr = np.array(truncate_matrix(Ul, 1, dimension))
                        del Ul
                    else:
                        del Ul
                        Utr = np.array(truncate_matrix(Ur, 1, dimension))
                        del Ur
                    self.courrent_node = ncon([Utr, Utr, self.courrent_node], [
                        [1, -3], [2, -4], [-1, -2, 1, 2]])
                    return self.courrent_node

                pass
            else:
                raise Exception("Invalid truncation direction supplied")
        else:
            raise Exception("direction must be string")
    def trace(self):
        self.factor = (self.factor)**4
        trace=ncon([self.courrent_node],[[1,1,2,2]]) 
        return trace*self.factor
    def directonal_reshape(self,direction):
        """Reshape after self cotraction"""
        """assumes xx x'x' yy'"""
        """or xx' yy y'y'"""
        if type(direction)==str:

            if len(self.courrent_node.shape) == 6:

                if direction == 'x':

                    if self.courrent_node.shape[0]==self.courrent_node.shape[1] \
                    and self.courrent_node.shape[2]==self.courrent_node.shape[3]:
                        self.reshape((self.courrent_node.shape[0]**2,\
                                      self.courrent_node.shape[2]**2,\
                                      self.courrent_node.shape[4],\
                                      self.courrent_node.shape[5]))

                    else:

                        raise Exception("(Proto)Wrong dimension for reshape")
        
                elif direction == 'y':

                    if self.courrent_node.shape[2]==self.courrent_node.shape[3] \
                    and self.courrent_node.shape[4]==self.courrent_node.shape[5]:

                        self.reshape((self.courrent_node.shape[0],\
                                      self.courrent_node.shape[1],\
                                      self.courrent_node.shape[2]**2,\
                                      self.courrent_node.shape[4]**2))

                    else:

                        raise Exception("(Proto)Wrong dimension for reshape")

                else:

                    raise Exception("(Proto) Invalid direction supplied, direction must be 'x' or 'y'")
                
            else:

                raise Exception("(Proto) Node not ready for directional reshape")

        else:

            raise Exception("direction must be string,'x' or 'y'")        
    def step(self, number_of_steps,dimesion):
        for i in range(0,number_of_steps):
            print(self.transformation_log+"\n\n")
            self.self_contract([-1,-3,-5,1],[-2,-4,1,-6])
            self.directonal_reshape('x')
            self.truncate('x',dimesion)
            self.self_contract([-1,1,-3,-5],[1,-2,-4,-6])
            self.directonal_reshape('y')
            self.truncate('y',dimesion)


"""Class implementation for HOTRG sweep"""
class HOTRG_sweep:

    def __init__(self,node,sweep_range,steps,dimension,output_path=""):
        self.node = node
        self.sweep_range = sweep_range
        self.steps = steps
        self.dimension = dimension
        if output_path =="":
            self.output_path = self.node.name + ".txt"
        else:
            self.output_path = output_path
    def log(self,param):
        with open(self.output_path,"a") as handle:
            handle.write(f"{param} {self.node.trace()}\n")
    
    def start(self):
        for val in self.sweep_range:
            self.node.renew(val)
            self.node.step(self.steps,self.dimension)
            self.log(val)

            

    




