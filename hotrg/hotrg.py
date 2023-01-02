import types 
import numpy as np 
from ncon import ncon 
import tensorly as tl
from hotrg.hotrg_tools.matrix import truncate_matrix
from time import perf_counter


class General_Node:
    

    def __init__(self,name,generator,*parameters ,verbose = 0):
        self.transformation_log =  ""
        self.factor = 0
        self.unit_step = 1
        self.name = name
        self.verbose = verbose 
        self.parameters = parameters 
        self.generate = generator 
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

    def factor_update(self,times):
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
        # Pre truncation adjustment
        self.truncate('x',dimesion)
        self.truncate('y',dimesion)
        for i in range(0,number_of_steps):
            self.self_contract([-1,-3,-5,1],[-2,-4,1,-6])
            self.directonal_reshape('x')
            self.factorize()
            self.truncate('x',dimesion)
            self.self_contract([-1,1,-3,-5],[1,-2,-4,-6])
            self.directonal_reshape('y')
            self.factorize()
            self.truncate('y',dimesion)

class Square_Node(General_Node):
    '''Square Node'''
    def truncate(self,direction_idx,dimension):
        if self.courrent_node.shape[direction_idx[0]]>dimension and self.courrent_node.shape[direction_idx[0]] == self.courrent_node.shape[direction_idx[1]]:
            unfolded_buffer = np.matrix(tl.unfold(self.courrent_node, direction_idx[0]))
            unfolded_buffer = unfolded_buffer @ unfolded_buffer.getH()
            U0, delta0, _ = np.linalg.svd(unfolded_buffer)

            del unfolded_buffer

            unfolded_buffer = unfolded_buffer = np.matrix(tl.unfold(self.courrent_node, direction_idx[1]))
            unfolded_buffer = unfolded_buffer @ unfolded_buffer.getH()
            U1, delta1, _ = np.linalg.svd(unfolded_buffer)

            del unfolded_buffer

            eps0, eps1 = 0, 0

            for i in range(dimension, len(delta0)):
                eps0 += delta0[i]
                eps1 += delta1[i]
            del delta0, delta1
            if eps0 < eps1:
                del U1
                Utr = np.array(truncate_matrix(U0, 1, dimension))
                del U0
            else:
                del U0
                Utr = np.array(truncate_matrix(U1, 1, dimension))
                del U1

            contr_pattern  = [val *-1 for val in list(range(1,len(self.courrent_node.shape)+1))]
            contr_pattern[direction_idx[0]]*=-1
            contr_pattern[direction_idx[1]]*=-1            
            self.courrent_node = ncon([Utr, Utr, self.courrent_node], [
                [direction_idx[0]+1, -(direction_idx[0]+1)], [direction_idx[1]+1, -(direction_idx[1]+1)], contr_pattern])
        return self.courrent_node


    def directional_reshape(self,direction):
        shape = self.courrent_node.shape
        if(direction == 'x'):
            self.reshape((shape[0]**2,shape[2]**2,shape[4]**2,shape[6]**2\
                        ,shape[8] ,shape[9],shape[10],shape[11]))
        if (direction == 'y'):
            self.reshape((shape[0],shape[1],shape[2],shape[3]\
                        ,shape[4]**2 ,shape[6]**2,shape[8]**2,shape[10]**2))
        return self.courrent_node

    def trace(self):
        trace = ncon([self.courrent_node],[[1,2,1,2,3,4,3,4]])
        return trace
    
    def step(self, number_of_steps, dimesion):
        for i in range(0,number_of_steps):
            self.self_contract([-1,-3,-5,-7,-9,-10,1,2],[-2,-4,-6,-8,1,2,-11,-12])
            self.directional_reshape('x')
            self.truncate((0,2),dimesion)
            self.truncate((1,3),dimesion)
            self.self_contract([-1,-2,1,2,-5,-7,-9,-11],[1,2,-3,-4,-6,-8,-10,-12])
            self.directional_reshape('y')
            self.truncate((4,6),dimesion)
            self.truncate((5,7),dimesion)




class HOTRG_sweep:
    """Class implementation for HOTRG sweep"""

    def __init__(self,node,sweep_range,steps,dimension,output_path=""):
        self.computed_names = list()
        self.computed_functions = list() 
        self.node = node
        self.side = np.sqrt(self.node.unit_step**steps)
        self.sweep_range = sweep_range
        self.steps = steps
        self.dimension = dimension
        if output_path =="":
            self.output_path = self.node.name + ".txt"
        else:
            self.output_path = output_path
        pass

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
    
    def calibrate(self,parameter,min_dim,max_dim,steps,output_dir):
        filename = f"{self.node.name}_S{steps}_P{parameter}_calibration.txt"
        header = f"Parameter: {parameter}\n" \
                 f"Min D: {min_dim}\n" \
                 f"Max D: {max_dim}\n" 
        header += " ".join(["D"] + self.computed_names)
        with open(output_dir+filename,"a") as handle:
            handle.write(header+"\n")
            handle.close()
        for D in range(min_dim,max_dim+1):
            self.node.renew(parameter)
            step_time = perf_counter()
            self.node.step(steps,D)
            print(self.node.transformation_log)
            print(f"Step Duration:{perf_counter()-step_time}\nD:{D}")
            log_str =f"{D}"
            for function in self.computed_functions:
                log_str += f" {function(parameter,self.node.trace(),self.node.factor)}"
            with open(output_dir+filename,"a") as handle:
                handle.write(log_str+"\n")
                handle.close()    



            

    




