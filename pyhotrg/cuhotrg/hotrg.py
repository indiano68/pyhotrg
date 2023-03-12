from ..hotrg import Cross_Node_Optimized
from  ..tools.matrix import truncate_matrix 
from ncon import ncon 
import numpy as np
import cupy as cp
from typing import Optional


class Cross_Node_GPU(Cross_Node_Optimized):
        
    def step_truncate(self,direction:str,dimension:int,t_method:Optional[str]=None)->None:

        if t_method is None or t_method == "SuperQ":
                if direction == 'x':

                    T_start_gpu = cp.asarray(self.courrent_node)
                    # Building Upper Q and it's SV 
                    A = cp.einsum("azxb,aqyb->xyzq",T_start_gpu,T_start_gpu)
                    B = cp.einsum("zaxb,qayb->xyzq",T_start_gpu,T_start_gpu)
                    tensorQu = cp.asnumpy(cp.einsum("xzab,yqab->xyzq",A,B))
                    Qshape =  tensorQu.shape
                    tensorQu = tensorQu.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                    # Building Upper Q and it's SV             
                    A = cp.einsum("azbx,aqby->xyzq",T_start_gpu,T_start_gpu)
                    B =cp.einsum("zabx,qaby->xyzq",T_start_gpu,T_start_gpu)
                    tensorQd = cp.asnumpy(cp.einsum("xzab,yqab->xyzq",A,B))
                    tensorQd = tensorQd.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))

                    SuperQ = tensorQu + tensorQd
                    U,_,_ = np.linalg.svd(SuperQ,hermitian=True)
                    Utr = truncate_matrix(U, 1, dimension)
                    Utr = Utr.reshape((Qshape[0],Qshape[1],dimension))

                    T_final_gpu = cp.zeros((self.courrent_node.shape[0],self.courrent_node.shape[1],dimension,dimension))
                    U_gpu = cp.asarray(Utr)
                    bound = self.courrent_node.shape[0]
                    for xf in range(0,bound):
                        B = cp.einsum("ijk,lim->klmj",U_gpu,T_start_gpu[xf,:,:,:])
                        C = cp.einsum("klmj,lqjt->kqmt",B,T_start_gpu)
                        T_final_gpu[xf,:,:,:]=cp.einsum("msc,kqms->qkc",U_gpu,C)
                    cp.cuda.Stream.null.synchronize()
                    self.courrent_node = cp.asnumpy(T_final_gpu)

                if direction == 'y':
                    T_start_gpu = cp.asarray(self.courrent_node)
                    # Building Left Qf and it's SV 
                    # tensorA = self.self_contract([-1,1,2,-3],[-2,1,2,-4])
                    A = cp.einsum("xabz,yabq->xyzq",T_start_gpu,T_start_gpu)
                    # tensorB = self.self_contract([-1,1,-3,2],[-2,1,-4,2])
                    B = cp.einsum("xazb,yaqb->xyzq",T_start_gpu,T_start_gpu)
                    # tensorQf = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]]) 
                    tensorQf =cp.asnumpy(cp.einsum("xzab,yqab->xyzq",A,B))
                    Qshape =  tensorQf.shape
                    tensorQf = tensorQf.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                    # Building Right Qb and it's SV 
                    # tensorA = self.self_contract([1,-1,2,-3],[1,-2,2,-4])
                    A = cp.einsum("axbz,aybq->xyzq",T_start_gpu,T_start_gpu)
                    # tensorB = self.self_contract([1,-1,-3,2],[1,-2,-4,2])
                    B = cp.einsum("axzb,ayqb->xyzq",T_start_gpu,T_start_gpu)
                    # tensorQb = ncon([tensorA,tensorB],[[-1,-3,1,2],[-2,-4,1,2]])
                    tensorQb = cp.asnumpy(cp.einsum("xzab,yqab->xyzq",A,B))
                    tensorQb = tensorQb.reshape((Qshape[0]*Qshape[1],Qshape[2]*Qshape[3]))
                    SuperQ = tensorQf + tensorQb
                    U,_,_ = np.linalg.svd(SuperQ,hermitian=True)
                    Utr = truncate_matrix(U, 1, dimension)
                    Utr = Utr.reshape((Qshape[0],Qshape[1],dimension))
                    bound = self.courrent_node.shape[2]
                    T_final_gpu = cp.zeros((dimension,dimension,self.courrent_node.shape[2],self.courrent_node.shape[3]))
                    U_gpu = cp.asarray(Utr)
                    #Recontructing T 
                    for yf in range(0,bound):
                        # tensorB2 = ncon([Utr,self.courrent_node[:,:,yf,:]],[[1,-4,-1],[1,-3,-2]])
                        # tensorC  = ncon([tensorB2,self.courrent_node],[[-1,1,-3,2],[2,-4,1,-2]])
                        # newTensor[:,:,yf,:] = ncon([Utr,tensorC],[[1,2,-2],[-1,-3,1,2]])          
                        B = cp.einsum("aqx,azy->xyzq",U_gpu,T_start_gpu[:,:,yf,:])
                        C = cp.einsum("xazb,bqay->xyzq",B,T_start_gpu)
                        T_final_gpu[:,:,yf,:]=cp.einsum("aby,xzab->xyz",U_gpu,C)
                    self.courrent_node =cp.asnumpy(T_final_gpu)
        else:
            raise NotImplementedError 

        self.transformation_log+=f"Step/Truncation in direction {direction} and D{dimension},new {self.courrent_node.shape}\n"






            

    




