from ..hotrg import Cross_Node_Optimized
from  ..tools.matrix import truncate_matrix 
from ncon import ncon 
import numpy as np
import cupy as cp
from typing import Optional


class Cross_Node_GPU(Cross_Node_Optimized):
    def truncate(self, direction:str, dimension:int):
        raise NotImplementedError
    
    def self_contract(self,order_1:list[int],order_2:list[int],update:Optional[bool] = False):

        if type(order_1)==type(order_2) and  type(order_1) is list:

            if len(order_1)==len(order_2) and len(order_1) == len(self.courrent_node.shape):

                newNode = ncon([self.courrent_node,self.courrent_node],[order_1,order_2])
                # Prototyping : New self contract does not update node
                # for el in order_1:
                #     if el>0: 
                #         self.factor_update(1)
                if update:
                    self.courrent_node = newNode
                    self.transformation_log+=f"Contraction with ordering {order_1} {order_2},new {self.courrent_node.shape}\n"
                return newNode
            else:
                raise Exception("(Proto) Invalid lenght of order1/2.")

        else:
            raise Exception("(Proto) self_contract params must be lists")

        
    def step_truncate(self,direction:str,dimension:int,t_method:Optional[str]=None)->None:

        if t_method is None or t_method == "SuperQ":
                if direction == 'x':
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
                    U,_,_ = np.linalg.svd(SuperQ)
                    Utr = truncate_matrix(U, 1, dimension)
                    Utr = Utr.reshape((Qshape[0],Qshape[1],dimension))
                    T_start_gpu = cp.asarray(self.courrent_node)
                    T_final_gpu = cp.zeros((self.courrent_node.shape[0],self.courrent_node.shape[1],dimension,dimension))
                    U_gpu = cp.asarray(Utr)
                    cp.cuda.Stream.null.synchronize()
                    bound = self.courrent_node.shape[0]
                    for xf in range(0,bound):
                        B = cp.einsum("ijk,lim->klmj",U_gpu,T_start_gpu[xf,:,:,:])
                        C = cp.einsum("klmj,lqjt->kqmt",B,T_start_gpu)
                        T_final_gpu[xf,:,:,:]=cp.einsum("msc,kqms->qkc",U_gpu,C)
                    cp.cuda.Stream.null.synchronize()
                    self.courrent_node = cp.asnumpy(T_final_gpu)

                if direction == 'y':

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
                    U,_,_ = np.linalg.svd(SuperQ)
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
        for step in range(0,number_of_steps):
            # x direction
            if(self.courrent_node.shape[2]**2<=dimension):
                self.self_contract([-1,1,-3,-5],[1,-2,-4,-6],update=True)
                self.directonal_reshape('y')
            else:
                self.step_truncate('x',dimension,t_method=t_method)
                pass 

            #y direction 

            if(self.courrent_node.shape[0]**2<=dimension):
                self.self_contract([-1,-3,-5,1],[-2,-4,1,-6],update=True)
                self.directonal_reshape('x')
            else:
                self.step_truncate('y',dimension,t_method=t_method)






            

    




