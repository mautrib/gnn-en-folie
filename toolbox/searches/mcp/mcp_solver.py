import os
import threading
import time
import torch
import numpy as np
import random
import toolbox.utils as utils

from string import ascii_letters,digits
NAME_CHARS = ascii_letters+digits

from numpy.ctypeslib import ndpointer
import ctypes

def pmc(ei,ej,nnodes,nnedges): #ei, ej is edge list whose index starts from 0
    degrees = np.zeros(nnodes,dtype = np.int32)
    new_ei = []
    new_ej = []
    for i in range(nnedges):
        degrees[ei[i]] += 1
        if ej[i] <= ei[i] + 1:
            new_ei.append(ei[i])
            new_ej.append(ej[i])
    maxd = max(degrees)
    offset = 0
    new_ei = np.array(new_ei,dtype = np.int32)
    new_ej = np.array(new_ej,dtype = np.int32)
    outsize = maxd
    output = np.zeros(maxd,dtype = np.int32)
    lib = ctypes.cdll.LoadLibrary("libpmc.so")
    fun = lib.max_clique
    #call C function
    fun.restype = np.int32
    fun.argtypes = [ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),ctypes.c_int32,
                  ctypes.c_int32,ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]
    clique_size = fun(len(new_ei),new_ei,new_ej,offset,outsize,output)
    max_clique = np.empty(clique_size,dtype = np.int32)
    max_clique[:]=[output[i] for i in range(clique_size)]

    return max_clique

def pmc_adj(adj):
    ei,ej = np.where(adj!=0)
    nnodes = adj.shape[0]
    nedges = len(ei)
    return pmc(ei, ej, nnodes, nedges)

class Thread_MCP_Solver(threading.Thread):
    def __init__(self, threadID, adj, name=''):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.adj = adj
        if name=='':
            name = ''.join(random.choice(NAME_CHARS) for _ in range(10))
        self.name = name
        self.solutions = []
        self.done=False
    
    def clear(self,erase_mode='all'):
        pass

    def run(self):
        #os.system(f"./mcp_solver.exe -v {self.fwname} >> {self.flname}")
        #self._read_adj()
        clique = pmc_adj(self.adj)
        self.solutions = [clique]
        self.done = True

class MCP_Solver():
    def __init__(self,adjs=None, max_threads=4, path='tmp_mcp/',erase_mode ='all'):
        utils.check_dir(path)
        self.path = path
        if adjs is None:
            self.adjs = []
        self.adjs = adjs
        assert max_threads>0, "Thread number put to 0."
        self.max_threads = max_threads
        self.threads = [None for _ in range(self.max_threads)]
        self.solutions  = []
        self.erase_mode=erase_mode
    
    @classmethod
    def from_data(adjs, **kwargs):
        return MCP_Solver(adjs, **kwargs)
    
    def load_data(self, adjs):
        self.adjs = adjs
    
    @property
    def n_threads(self):
        return np.sum([thread is not None for thread in self.threads])
    
    def no_threads_left(self):
        return np.sum([thread is None for thread in self.threads])==self.max_threads
    
    def is_thread_available(self,i):
        return self.threads[i] is None
    
    def clean_threads(self):
        for i,thread in enumerate(self.threads):
            if thread is not None and thread.done:
                id = thread.threadID
                print(f"Solution {id} on thread {i} is done.")
                self.solutions[id] = thread.solutions
                thread.clear(erase_mode=self.erase_mode)
                self.threads[i] = None
    
    def reset(self,bs):
        self.solutions = [list() for _ in range(bs)]
        self.threads = [None for _ in range(self.max_threads)]
    
    def solve(self):
        exp_name = ''.join(random.choice(NAME_CHARS) for _ in range(10))

        solo = False
        if isinstance(self.adjs, torch.Tensor):
            adjs = self.adjs.detach().clone()
        elif isinstance(self.adjs, list):
            bs = len(self.adjs)
            adj_shape = self.adjs[0].shape
            adjs = torch.zeros((bs,)+adj_shape)
            for i in range(bs):
                adjs[i] = self.adjs[i]
        else:
            adjs = self.adjs
        if len(adjs.shape)==2:
            solo = True
            adjs = adjs.unsqueeze(0)
        bs,n,_ = adjs.shape
        self.reset(bs)

        counter = 0
        while counter<bs or not self.no_threads_left():
            for thread_slot in range(self.max_threads):
                if counter < bs and self.is_thread_available(thread_slot):
                    adj = adjs[counter]
                    new_thread = Thread_MCP_Solver(counter,adj,name=os.path.join(self.path,f'tmp-mcp-{counter}-{exp_name}'))
                    #print(f"Putting problem {counter} on thread {thread_slot}")
                    self.threads[thread_slot] = new_thread
                    new_thread.start()
                    counter+=1
            self.clean_threads()
        


if __name__=='__main__':
    def test_mcp_solver(bs,n,max_threads=4):
        adjs = torch.empty((bs,n,n)).uniform_()
        adjs = (adjs.transpose(-1,-2)+adjs)/2
        adjs = (adjs<(0.5)).to(int)
        mcp_solver = MCP_Solver(adjs,max_threads)
        mcp_solver.solve()
        clique_sols = mcp_solver.solutions
        return clique_sols
    
    n=100
    t0 = time.time()
    [test_mcp_solver(10,n,max_threads=4) for _ in range(10)]
    print("Time taken :", time.time()-t0)
