import types 
import numpy as np 
from ncon import ncon 
import tensorly as tl
from pyhotrg.tools.matrix import truncate_matrix
from time import perf_counter
from typing import Callable,Optional, Tuple


class General_Node:
    
    '''Abstract class that generaly defines an hypercubic tensor network'''
    def __init__(self,name:str, generator: Callable[..., np.ndarray],*parameters ,verbose: Optional[bool] = False)->None:
        self.transformation_log : str =  "" #log of transformation made to the fundamental tensor
        self.factor : float = 0 # place older for tensor factorization
        self.unit_step :int = 1 # of lattices contracted upun network renormalization
        self.name = name # a lable for the network
        self.verbose = verbose # needs to be furter implemented
        self.parameters = parameters # needs to be furter implemented
        self.generate = generator # function that returns the fundamental tensor of the network for different parameters
        self.courrent_node:np.ndarray = self.generate(*self.parameters) # the fundamental tensor 
        self.transformation_log+=f"Node Generated with parameter\\s {self.parameters} \n"

    def renew(self,*parameters)->None:
        '''Regenerate the fundamental tensor of the network with new parameters'''
        self.parameters = parameters
        self.original_node = self.generate(*self.parameters)
        self.courrent_node = self.generate(*self.parameters)
        self.factor = 0
        self.transformation_log = f"Node renewed with parameter\\s {self.parameters} \n"
    
        
    def self_contract(self,order_1:list[int],order_2:list[int])->np.ndarray:
        '''Handles the contraction of the Fundamental tensor with himself along a direction'''
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

    def reshape(self,dim_tuple: Tuple[int,...])->np.ndarray:
        '''Reshape the fundamental tensor'''
        if type(dim_tuple) == tuple:
            self.courrent_node = self.courrent_node.reshape(dim_tuple)
            self.transformation_log+=f"Rehsaping with dimentions {dim_tuple},new {self.courrent_node.shape}\n"
            return self.courrent_node

        else:

            raise Exception("Argument must be tuple.")
    def trace(self)->float:
        raise NotImplementedError
    def step(self, number_of_steps:int,dimesion:int,t_method: Optional[str]=None)->None:
        raise NotImplementedError
    def factor_update(self,times:int):
        raise NotImplementedError


class Cross_Node(General_Node):

    '''Naive Hotrg algorithm for 2-dimensional equilateral hypercubic tenso network''' 
    unit_step = 4 
    def truncate(self,direction: str, dimension:int):

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
        #if (self.courrent_node[0, 0, 0, 0] > 1e+2 or self.courrent_node[0, 0, 0, 0] < 1e-2):
        f_norm = np.linalg.norm(self.courrent_node)
        if (f_norm > 1e+5):
            # toString = np.format_float_scientific(self.courrent_node[0, 0, 0, 0])
            toString = np.format_float_scientific(f_norm)
            e = toString.find("e")
            newFactor = float(toString[e+1:])
            self.factor+= newFactor
            self.courrent_node = self.courrent_node/(10**newFactor)
            self.transformation_log+=f"Factorized: NewFactor e{toString[e+1:]},Total {self.factor}\n"

    def factor_update(self,times:int):
        # for index in range(0,len(self.factor)):
        #     self.factor[index]*=2
        for i in range(0,times):
            self.factor*=2
        self.transformation_log+=f"Factor Updated, New: {self.factor}\n"       
    
    def directonal_reshape(self,direction:str):
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

    def step(self, number_of_steps:int,dimesion:int,t_method: Optional[str]=None)->None:
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
    '''Prototype...'''
    def truncate(self,direction_idx:Tuple[int, int],dimension:int)->np.ndarray:
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


    def directional_reshape(self,direction:str)->np.ndarray:
        shape = self.courrent_node.shape
        if(direction == 'x'):
            self.reshape((shape[0]**2,shape[2]**2,shape[4]**2,shape[6]**2\
                        ,shape[8] ,shape[9],shape[10],shape[11]))
        if (direction == 'y'):
            self.reshape((shape[0],shape[1],shape[2],shape[3]\
                        ,shape[4]**2 ,shape[6]**2,shape[8]**2,shape[10]**2))
        return self.courrent_node

    def trace(self)->float:
        trace = ncon([self.courrent_node],[[1,2,1,2,3,4,3,4]])[0]
        return trace
    
    def step(self, number_of_steps: int, dimesion: int,t_method: Optional[str]=None)->None:
        for i in range(0,number_of_steps):
            self.self_contract([-1,-3,-5,-7,-9,-10,1,2],[-2,-4,-6,-8,1,2,-11,-12])
            self.directional_reshape('x')
            self.truncate((0,2),dimesion)
            self.truncate((1,3),dimesion)
            self.self_contract([-1,-2,1,2,-5,-7,-9,-11],[1,2,-3,-4,-6,-8,-10,-12])
            self.directional_reshape('y')
            self.truncate((4,6),dimesion)
            self.truncate((5,7),dimesion)


class Cross_Node_Optimized(Cross_Node):
    '''Optimazed HOTRG algorithm for 2dimensional Hypercubic tensor networks as described in the thesis,alg.4'''
    def truncate(self, direction:str, dimension:int):
        raise NotImplementedError
    
    def self_contract(self,order_1:list[int],order_2:list[int],update:Optional[bool] = False):

        if type(order_1)==type(order_2) and  type(order_1) is list:

            if len(order_1)==len(order_2) and len(order_1) == len(self.courrent_node.shape):

                newNode = ncon([self.courrent_node,self.courrent_node],[order_1,order_2])
                if update:
                    self.courrent_node = newNode
                    self.transformation_log+=f"Contraction with ordering {order_1} {order_2},new {self.courrent_node.shape}\n"
                return newNode
            else:
                raise Exception("(Proto) Invalid lenght of order1/2.")

        else:
            raise Exception("(Proto) self_contract params must be lists")

        
    def step_truncate(self,direction:str,dimension:int,t_method:Optional[str]=None)->None:
        if t_method is None or t_method == "fb":
            '''Like algorithm 2 and 3 but with xie truncation'''
            if direction == 'x':
                # Building Upper U and it's SV 
                tensorA = self.self_contract([1,-3,-1,2],[1,-4,-2,2])
                tensorB = self.self_contract( [-3,1,-1,2],[-4,1,-2,2])
                tensorQ = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]])
                Qshape =  tensorQ.shape
                tensorQ = tensorQ.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                Uu, deltaU, _ = np.linalg.svd(tensorQ)

                # Building Upper U and it's SV             
                tensorA = self.self_contract([1,-3,2,-1],[1,-4,2,-2])
                tensorB = self.self_contract( [-3,1,2,-1],[-4,1,2,-2])
                tensorQ = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]])
                #Qshape =  tensorQ.shape
                tensorQ = tensorQ.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                Ud, deltaD, _ = np.linalg.svd(tensorQ)
                if sum(deltaU[dimension:-1])<sum(deltaD[dimension:-1]):
                    Utr = np.array(truncate_matrix(Uu, 1, dimension))
                else:
                    Utr = np.array(truncate_matrix(Ud, 1, dimension))
                Utr = Utr.reshape((Qshape[0],Qshape[1],dimension))
                tensorB2 = ncon([Utr,self.courrent_node],[[1,-5,-2],[-1,-3,1,-4]]) 
                tensorC = ncon([tensorB2,self.courrent_node],[[-1,-2,1,-4,2],[1,-3,2,-5]])
                newTensor = ncon([Utr,tensorC],[[1,2,-4],[-1,-3,-2,1,2]])
                self.courrent_node = newTensor

            if direction == 'y':

                # Building Left U and it's SV 
                tensorA = self.self_contract([-1,1,2,-3],[-2,1,2,-4])
                tensorB = self.self_contract([-1,1,-3,2],[-2,1,-4,2])
                tensorQ = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]]) 
                Qshape =  tensorQ.shape
                tensorQ = tensorQ.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                Ul, deltaL, _ = np.linalg.svd(tensorQ)
                # Building Right U and it's SV 
                tensorA = self.self_contract([1,-1,2,-3],[1,-2,2,-4])
                tensorB = self.self_contract([1,-1,-3,2],[1,-2,-4,2])
                tensorQ = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]])

                Qshape =  tensorQ.shape
                tensorQ = tensorQ.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                Ur, deltaR, _ = np.linalg.svd(tensorQ)
                if sum(deltaR[dimension:-1])<sum(deltaL[dimension:-1]):
                    Utr = np.array(truncate_matrix(Ur, 1, dimension))
                else:
                    Utr = np.array(truncate_matrix(Ul, 1, dimension))
                Utr = Utr.reshape((Qshape[0],Qshape[1],dimension))

                #Recontructing T 
                tensorB2 = ncon([Utr,self.courrent_node],[[1,-5,-1],[1,-4,-2,-3]])
                tensorC  = ncon([tensorB2,self.courrent_node],[[-1,-2,1,-4,2],[2,-5,1,-3]])
                newTensor = ncon([Utr,tensorC],[[1,2,-2],[-1,-3,-4,1,2]])          
                self.courrent_node =newTensor

        elif t_method == "SuperQ":
                if direction == 'x':
                    '''Algorithm 2'''
                    # Building Upper Q and it's SV 
                    tensorA = self.self_contract([1,-3,-1,2],[1,-4,-2,2])
                    tensorB = self.self_contract( [-3,1,-1,2],[-4,1,-2,2])
                    tensorQu = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]])
                    Qshape =  tensorQu.shape
                    tensorQu = tensorQu.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                    # Building Upper Q and it's SV             
                    tensorA = self.self_contract([1,-3,2,-1],[1,-4,2,-2])
                    tensorB = self.self_contract( [-3,1,2,-1],[-4,1,2,-2])
                    tensorQd = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]])
                    tensorQd = tensorQd.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                    SuperQ = tensorQu +tensorQd
                    U,_,_ = np.linalg.svd(SuperQ,hermitian=True)
                    Utr = truncate_matrix(U, 1, dimension)
                    Utr = Utr.reshape((Qshape[0],Qshape[1],dimension))
                    tensorB2 = ncon([Utr,self.courrent_node],[[1,-5,-2],[-1,-3,1,-4]]) 
                    tensorC = ncon([tensorB2,self.courrent_node],[[-1,-2,1,-4,2],[1,-3,2,-5]])
                    newTensor = ncon([Utr,tensorC],[[1,2,-4],[-1,-3,-2,1,2]])
                    self.courrent_node = newTensor

                if direction == 'y':
                    '''Algorithm 3'''
                    # Building Left Qf and it's SV 
                    tensorA = self.self_contract([-1,1,2,-3],[-2,1,2,-4])
                    tensorB = self.self_contract([-1,1,-3,2],[-2,1,-4,2])
                    tensorQf = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]]) 
                    Qshape =  tensorQf.shape
                    tensorQf = tensorQf.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                    # Building Right Qb and it's SV 
                    tensorA = self.self_contract([1,-1,2,-3],[1,-2,2,-4])
                    tensorB = self.self_contract([1,-1,-3,2],[1,-2,-4,2])
                    tensorQb = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]])
                    tensorQb = tensorQb.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                    SuperQ = tensorQf + tensorQb
                    U,_,_ = np.linalg.svd(SuperQ, hermitian= True)
                    Utr = truncate_matrix(U, 1, dimension)
                    Utr = Utr.reshape((Qshape[0],Qshape[1],dimension))
                    #Recontructing T 
                    tensorB2 = ncon([Utr,self.courrent_node],[[1,-5,-1],[1,-4,-2,-3]])
                    tensorC  = ncon([tensorB2,self.courrent_node],[[-1,-2,1,-4,2],[2,-5,1,-3]])
                    newTensor = ncon([Utr,tensorC],[[1,2,-2],[-1,-3,-4,1,2]])          
                    self.courrent_node =newTensor
        else:
            raise NotImplementedError 

        self.transformation_log+=f"Step/Truncation in direction {direction} and D{dimension},new {self.courrent_node.shape}\n"

    def step(self, number_of_steps: int, dimension:int,t_method: Optional[str]=None):
        '''Algorithm 4'''
        for step in range(0,number_of_steps):
            # x direction
            if(self.courrent_node.shape[2]**2<=dimension):
                self.self_contract([-1,1,-3,-5],[1,-2,-4,-6],update=True)
                self.directonal_reshape('y')
            else:
                self.step_truncate('x',dimension,t_method=t_method)
                pass 
            self.factor_update(1)
            #y direction 
            if(self.courrent_node.shape[0]**2<=dimension):
                self.self_contract([-1,-3,-5,1],[-2,-4,1,-6],update=True)
                self.directonal_reshape('x')
            else:
                self.step_truncate('y',dimension,t_method=t_method)
            self.factor_update(1)
            self.factorize()



class HOTRG_sweep:
    """Class implementation for HOTRG sweep"""

    def __init__(self,node:General_Node,sweep_range:list[float],steps:int,dimension:int,t_method:Optional[str]=None,output_path:str=""):
        self.computed_names:list[str] = list()
        self.computed_functions:list[Callable] = list() 
        self.node:General_Node = node
        self.side:float = np.sqrt(self.node.unit_step**steps)
        self.sweep_range:list[float] = sweep_range
        self.steps: int = steps
        self.dimension:int = dimension
        self.t_method = t_method
        if output_path =="":
            self.output_path = self.node.name + ".txt"
        else:
            self.output_path = output_path
        pass

    def _log_header(self,method:Optional[str]=None):
        if method is None:
            header = ["Parameter","Trace","Factor(e)"]
        else:
            header = ["Parameter",method]
        header += self.computed_names
        with open(self.output_path,"a") as handle:
            handle.write(" ".join(header)+"\n")
            handle.close()
        pass 

    def _log_data(self,param:float ,val:float, method: Optional[str]=None):
        if method is None: 
            log_str =f"{param} {val} {int(self.node.factor)}"
        else:
            log_str =f"{param} {val}"

        for function in self.computed_functions:
            log_str += f" {function(param,val,self.node.factor)}"

        with open(self.output_path,"a") as handle:
            # handle.write(f"{param} {self.node.trace()}E{int(self.node.factor)}\n")
            handle.write(log_str+"\n")
            handle.close()

    def add_to_compute(self,name:str,function:Callable[[float,float,float],float]):
        if type(name) is str and type(function) is types.FunctionType:
            self.computed_names.append(name)
            self.computed_functions.append(function)
        else:
            raise Exception("(Proto) wrong arguments passed")

    def compute_logZ(self): 
        sites = self.node.unit_step ** self.steps
        return (np.log(self.node.trace())+self.node.factor*np.log(10))/(sites)


    def start(self,method: Optional[str] = None, delta: Optional[float]=None):
        self._log_header(method=method)
        sweep_time = perf_counter()
        if method != "fd":
            for param in self.sweep_range:
                self.node.renew(param)
                step_time = perf_counter()
                self.node.step(self.steps,self.dimension,t_method=self.t_method)
                print(self.node.transformation_log)
                print(f"Step Duration:{perf_counter()-step_time}")
                if method is None:
                    self._log_data(param,self.node.trace())
                if method == "lnZ":
                    self._log_data(param,self.compute_logZ(),method=method)
        else:
            if delta is None:
                delta = 0.001
            for param in self.sweep_range:
                val_1= param-delta/2
                val_2= param+delta/2
                self.node.renew(val_1)
                step_time = perf_counter()
                self.node.step(self.steps,self.dimension,t_method=self.t_method)
                res_1 = self.compute_logZ()
                print(self.node.transformation_log)
                print()
                self.node.renew(val_2)
                self.node.step(self.steps,self.dimension,t_method=self.t_method)
                res_2 = self.compute_logZ()
                print(self.node.transformation_log)
                print()
                print(f"Step Duration:{perf_counter()-step_time}")
                self._log_data(param,(res_2-res_1)/delta,method=method)


        print(f"Sweep Duration:{perf_counter()-sweep_time}")
    
    
    def calibrate(self,parameter:float,min_dim:int,max_dim:int,steps:int,output_dir:str):
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
            # for function in self.computed_functions:
            #     log_str += f" {function(parameter,self.node.trace(),self.node.factor)}"
            log_str += f" {self.compute_logZ()}"
            with open(output_dir+filename,"a") as handle:
                handle.write(log_str+"\n")
                handle.close()    



            

    




