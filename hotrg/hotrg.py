import types 
import numpy as np 
from ncon import ncon 
import tensorly as tl
from hotrg_tools.matrix import truncate_matrix
from time import perf_counter


class General_Node:
    
    transformation_log =  ""
    factor = 0
    unit_step = 1

    def __init__(self,name,generator,*parameters ,verbose = 0):
        self.name = name
        self.verbose = verbose 
        self.parameters = parameters 
        self.generate = generator 
        self.original_node = self.generate(*self.parameters)
        self.courrent_node = self.generate(*self.parameters)
        self.transformation_log+=f"Node Generated with parameter\s {self.parameters} \n"
        pass

    def renew(self,*parameters):
        self.parameters = parameters
        self.original_node = self.generate(*self.parameters)
        self.courrent_node = self.generate(*self.parameters)
        self.factor = 0
        self.transformation_log = f"Node renewed with parameter\s {self.parameters} \n"
    
        
    def self_contract(self,order_1,order_2):

        if type(order_1)==type(order_2) and  type(order_1) is list:

            if len(order_1)==len(order_2) and len(order_1) == len(self.courrent_node.shape):

                self.courrent_node = ncon([self.courrent_node,self.courrent_node],[order_1,order_2])

                for el in order_1:
                    if el>0: 
                        self.factor_update(1)
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

    def factor_update(self):
        pass

    def factor_compute(self):
        pass

    def trace(self):
        pass 




class Cross_Node(General_Node):
    """Expand general node to allow Xie Type Step and Truncation"""
    """For nodes with for legs of equal order""" 
    unit_step = 4 
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

        trace=ncon([self.courrent_node],[[1,1,2,2]]) 
        return trace 


    def factorize(self):
        # Prototype for testing 
        # Factor out just powers of ten, approach must be discussed
        if (self.courrent_node[0, 0, 0, 0] > 1e+2 or self.courrent_node[0, 0, 0, 0] < 1e-2):

            toString = np.format_float_scientific(self.courrent_node[0, 0, 0, 0])
            e = toString.find("e")
            newFactor = float(toString[e+1:])
            self.factor+= newFactor
            self.courrent_node = self.courrent_node/(10**newFactor)
            self.transformation_log+=f"Factorized: NewFactor e{toString[e+1:]},Total {self.factor}\n"

    def factor_update(self,times):
        # for index in range(0,len(self.factor)):
        #     self.factor[index]*=2
        for i in range(0,times):
            self.factor*=2
        self.transformation_log+=f"Factor Updated, New: {self.factor}\n"       
    
    # def factor_compute(self):
    #     return sum(self.factor)

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
            self.self_contract([-1,-3,-5,1],[-2,-4,1,-6])
            self.directonal_reshape('x')
            self.factorize()
            self.truncate('x',dimesion)
            self.self_contract([-1,1,-3,-5],[1,-2,-4,-6])
            self.directonal_reshape('y')
            self.factorize()
            self.truncate('y',dimesion)


class HOTRG_sweep:
    """Class implementation for HOTRG sweep"""
    computed_names = list()
    computed_functions = list() 

    def __init__(self,node,sweep_range,steps,dimension,output_path=""):
        self.node = node
        self.side = np.sqrt(self.node.unit_step**steps)
        self.sweep_range = sweep_range
        self.steps = steps
        self.dimension = dimension
        if output_path =="":
            self.output_path = self.node.name + ".txt"
        else:
            self.output_path = output_path

    def _log_header(self):
        header = ["Parameter","Trace","Factor(e)"] + self.computed_names
        with open(self.output_path,"a") as handle:
            handle.write(" ".join(header)+"\n")
            handle.close()
        pass 

    def _log_data(self,param):
        log_str =f"{param} {self.node.trace()} {int(self.node.factor)}"
        for function in self.computed_functions:
            log_str += f" {function(param,self.node.trace(),self.node.factor)}"
        with open(self.output_path,"a") as handle:
            # handle.write(f"{param} {self.node.trace()}E{int(self.node.factor)}\n")
            handle.write(log_str+"\n")
            handle.close()
    

    def add_to_compute(self,name,function):
        if type(name) is str and type(function) is types.FunctionType:
            self.computed_names.append(name)
            self.computed_functions.append(function)
        else:
            raise Exception("(Proto) wrong arguments passed")



    def start(self):
        self._log_header()
        sweep_time = perf_counter()
        for val in self.sweep_range:
            self.node.renew(val)
            step_time = perf_counter()
            self.node.step(self.steps,self.dimension)
            print(self.node.transformation_log)
            print(f"Step Duration:{perf_counter()-step_time}")
            self._log_data(val)
        print(f"Sweep Duration:{perf_counter()-sweep_time}")

            

    




