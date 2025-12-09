import ctypes 
#import learn_ising 
from learn_ising import *
#import torch.serialization
#torch.serialization.weights_only = False



import torch
import torch
###########à
# Salvo la funzione originale
_orig_load = torch.load

# Creo una versione modificata che forza weights_only=False
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)

# Rimpiazzo torch.load globalmente
torch.load = _patched_load


#######
from torch.nn.functional import one_hot
from torch_geometric.datasets import WikipediaNetwork, WebKB, Actor, Planetoid,Amazon,Coauthor,WikiCS
from graphesn import StaticGraphReservoir, initializer, Readout
from graphesn.util import graph_spectral_norm
import warnings 
import numpy as np
import subprocess
import itertools
from argparse import ArgumentParser
import sys
import os
warnings.filterwarnings('ignore')
import time 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

from scipy.special import digamma
from sklearn.metrics import pairwise_distances
import torch.nn as nn 
import torch.optim as optim 
import random 
import copy 
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import ToUndirected



def get_dataset(root, name): 
    if name in ['chameleon','squirrel']: 
        return WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True)
    elif name in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root=root, name=name)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=root, name=name, split='geom-gcn')
    elif name in ['CS','Physics']: 
        return Coauthor(root=root,name=name)
    elif name in ["Photo","Computers"]: 
        return Amazon(root=root, name=name)
    elif name=='WikiCS': 
        return WikiCS(root=root) 
    elif name == 'actor':
        return Actor(root=root)
    elif name == 'arxiv': 
        return PygNodePropPredDataset(name="ogbn-arxiv", transform=ToUndirected())
    else:
        raise ValueError(f'Unknown dataset `{name}`')
        

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
class Node_Classification(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=10): 
        super().__init__() 
        self.fc1 = nn.Linear(input_dim,hidden_dim) #,bias=False) 
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
        
    def _init_weights(self): 
        for m in self.modules():
            if isinstance(m,nn.Linear): 
                if m is self.fc2: 
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias) 
                else: 
                    nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
                    nn.init.zeros_(m.bias) 
    def forward(self,x): 
        x = self.fc1(x) 
        x=self.norm(x) 
        x=self.activation(x) 
        return self.fc2(x) 



class Node_Classification_2(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
               # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.fc1(x)
        x=self.norm(x)
        x = self.relu(x)
        return self.fc2(x)



        
        
        
def mlp_test(dataset,data,emb,dimension,name_file):
    device='cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   # print(device) 
    scaler = StandardScaler()
    split_ = np.load(name_file)
    #y = one_hot(dmlp_test(dataset,data,emb,dimension,name_file):
    y =data.y #one_hot(data.y, dataset.num_classes).float()
    scaler = StandardScaler()   #ata.y, dataset.num_classes).float()
    #x= scaler.fit_transform(emb)
    scaler = StandardScaler().fit(emb[split_["train"]]) 
    x = scaler.transform(emb)    #transform in all the data 
    
    x = torch.Tensor(x).to(device) 
    train_x = x[split_["train"]].to(device)
    train_y = y[split_["train"]].to(device) 
    val_x = x[split_["valid"]].to(device) 
    val_y = y[split_["valid"]].to(device) 
    #test_x = x[split_["test"]].to(device) 
    #test_y = y[split_["test"]].to(device)
    lr_ = [0.01,0.005,0.001]
    best_acc=0
    acc = 0
    epochs=500
    checkpoint = None 

    for lr in lr_:
        set_seed(21)
        model=Node_Classification(input_dim=dimension,hidden_dim=512,output_dim=dataset.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=lr)
        for epoch in range(epochs):
            model.train()
            total_loss=0
            optimizer.zero_grad()
            out = model(train_x)
            loss = criterion(out,train_y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = model(x)
                acc = ((y_pred[split_["valid"]].argmax(dim=-1).to('cpu') == data.y[split_["valid"]]).float().to('cpu').mean() * 100).item()
            if acc> best_acc:
                best_acc = acc
                checkpoint = deepcopy(model.state_dict())
    model.load_state_dict(checkpoint) 
    model.eval() 
    with torch.no_grad():
        y_pred = model(x)
        test_acc = ((y_pred[split_["test"]].argmax(dim=-1).to('cpu') == data.y[split_["test"]]).float().to('cpu').mean() * 100).item()
    
    
    return best_acc,test_acc # checkpoint 
            
             
    
        
        
    
            
def standard_regression(dataset,data,emb,dimension,name_file):
    print("Start standard regresion") 
    device= 'cpu' 
  #  aa=1
   # if aa==1:
    #    return 0
    split_ = np.load(name_file)
    y = one_hot(data.y, dataset.num_classes).float().to(device)
    x = torch.Tensor(emb).to(device) 
    
    
    ld_list = ld_list=np.linspace(0.01, 25, 20)
    
    score_test = [] 
    score_val = []

    
    for ld in ld_list: 
        readout = Readout(num_features=dimension,num_targets=dataset.num_classes).to(device)
        
        #split_["train"] , #split["valid"] #split["test"] 
        
        readout.fit((x[split_["train"]],y[split_["train"]]),ld)
        y_pred = readout(x.to(device))
        score_test.append((y_pred[split_["test"]].argmax(dim=-1).to('cpu') == data.y[split_["test"]]).float().to('cpu').mean() * 100)
        score_val.append((y_pred[split_["valid"]].argmax(dim=-1).to('cpu') == data.y[split_["valid"]]).float().to('cpu').mean() * 100)

   # best_acc = max(score_val) 
        
    best_idx = max(range(len(score_val)), key=lambda i: score_val[i])
 
    best_acc = score_val[best_idx]
    best_test_acc = score_test[best_idx]               

 
    return best_acc.item(),best_test_acc.item()       
        


    
def simple_regression(dataset,data,emb,dimension):


    device = 'cpu'
    #ld = 0
    y = one_hot(data.y, dataset.num_classes).float().to(device)
    x = torch.Tensor(emb).to(device)

    best_acc_test = 0
    bast_acc_val = 0

    test_acc_list =[]
    val_acc_list = []
    std_val_acc = [] 
    std_test_acc = []

    #readout = Readout(num_features=dimension,num_targets=dataset.num_classes).to(device)





    ld_list = ld_list=np.linspace(0.01, 25, 20)

    for ld in ld_list:
        readout = Readout(num_features=4096,num_targets=dataset.num_classes).to(device)
        lista_score_train = []

        lista_score_val = []

        lista_score_test = []

        

        for i in range(data.train_mask.shape[1]):
            readout.fit((x[data.train_mask[:,i]], y[data.train_mask[:,i]]), ld)
            y_pred = readout(x.to(device))
            lista_score_val.append((y_pred[data.val_mask[:,i]].argmax(dim=-1).to('cpu') == data.y[data.val_mask[:,i]]).float().to('cpu').mean() * 100)
            lista_score_test.append((y_pred[data.test_mask[:,i]].argmax(dim=-1).to('cpu') == data.y[data.test_mask[:,i]]).float().to('cpu').mean() * 100)

        lista_score_test = np.array(lista_score_test)
        lista_score_val = np.array(lista_score_val)
        mean_test = np.mean(lista_score_test)
        mean_val = np.mean(lista_score_val)
        test_acc_list.append(mean_test)
        val_acc_list.append(mean_val)
        std_val_acc.append(np.std(lista_score_val)) 
        std_test_acc.append(np.std(lista_score_test))

    #best_acc = max(test_acc_list)

    #return best_acc

    best_idx = max(range(len(val_acc_list)), key=lambda i: val_acc_list[i])
    best_val = val_acc_list[best_idx]
    best_test = test_acc_list[best_idx]
    std_val = std_val_acc[best_idx]
    std_test =std_test_acc[best_idx]

    return best_val, best_test,std_val,std_test







def simple_regression_2(dataset,data,emb,dimension): 


    device = 'cpu' 
    #ld = 0 
    y = one_hot(data.y, dataset.num_classes).float().to(device)
    x = torch.Tensor(emb).to(device)
    
    best_acc_test = 0
    bast_acc_val = 0 
    
    test_acc_list =[]
    val_acc_list = [] 
    
    #readout = Readout(num_features=dimension,num_targets=dataset.num_classes).to(device)
    
   
    
    
    
    ld_list = ld_list=np.linspace(0.01, 25, 20)
    
    for ld in ld_list:
        readout = Readout(num_features=4096,num_targets=dataset.num_classes).to(device)
        lista_score_train = [] 
    
        lista_score_val = [] 
    
        lista_score_test = [] 
    
        
    
        for i in range(10): 
            readout.fit((x[data.train_mask[:,i]], y[data.train_mask[:,i]]), ld)
            y_pred = readout(x.to(device))
            lista_score_test.append((y_pred[data.test_mask[:,i]].argmax(dim=-1).to('cpu') == data.y[data.test_mask[:,i]]).float().to('cpu').mean() * 100)
        
        lista_score_test = np.array(lista_score_test)
        mean_test = np.mean(lista_score_test) 
   
        test_acc_list.append(mean_test) 
        
    best_acc = max(test_acc_list)  
    
    return best_acc
    
    

def main(args): 

    np.random.seed(0) 
    
    data_ = str(args.dataset) 
    
    dataset = get_dataset(root='./Dataset___',name=data_)
    
    data = dataset[0] 
    
    dimension = int(args.dimension) 
    
    matrix_type = int(args.w_type) 
    
    w_connections = int(args.w_connections)
    
    save_dir = data_ +"_sparsity_"
    
    dir_ = "bias_"+str(args.bias)+"_"+str(matrix_type)+"_"+str(dimension)+"_"+str(w_connections)+"_"+str(args.random_state_w)
    
    tot_dir = os.path.join(save_dir,dir_)    #directory dove dalva i dati 
    
    if not os.path.exists(tot_dir):
        os.makedirs(tot_dir) 

    file_with_acc = os.path.join(tot_dir,"accuracy.txt") 
    file_ridge_test = os.path.join(tot_dir,"ridge_test_accuracy.txt")
   # file_with_acc = os.path.join(tot_dir,"accuracy.txt") 
    file_with_mlP_acc = os.path.join(tot_dir,"mlp_accuracy.txt")   
    file_with_test_mlp =os.path.join(tot_dir,"test_accuracy.txt")    
    energy_file =os.path.join(tot_dir,"sim_energy.txt") 
    file_with_val_std = os.path.join(tot_dir,"std_val.txt") 
    file_with_test_std = os.path.join(tot_dir,"std_test.txt")

    #file_with_time = "timing"+str(args.n_threads)+".txt"

    adj_matrix_name = "Adj_"+data_+".txt"
    adj_matrix_name =adj_matrix_name.encode('utf-8')
    
    model = Ising(
        data, 
        adj_matrix_name,
        data.num_nodes,
        args.steps,
        args.temperature,
        w_connections,
        args.random_state_w,
        args.random_state_iteration,
        dimension,
        args.bias,
        nthreads=args.n_threads,
        matrix_type=matrix_type,
        continue_=args.continue_,
        root_=tot_dir,
        data_=data_)
        
        
        
        
        
        
        
        
        
 
    
    print(model) 
    
           
    
    
          #model=Ising(data,adj_matrix_name,data.num_nodes,args.steps0,w_connections,args.random_state_w,args.random_state_iteration,dimension,args.bias,nthreads=args.n_threads,matrix_type=matrix_type,continue_=arg>
   # model.init_embedding()    
    print(model) 
    _,emb = model.save_to_numpy(save_file=False)    
    split_file = data_+".npz"
    name_file = os.path.join("data_split",split_file)
    #test_accuracy = standard_regression(dataset,data,emb,dimension,name_file)
    mlp_test_accuracy = 0 
    mlp_val_accuracy=0
#    mlp_val_accuracy,mlp_test_accuracy =mlp_test(dataset,data,emb,dimension,name_file)  #provvisori prova 
    print("First run mlp acuracy = ",mlp_test_accuracy)
    num_loop = int(args.steps/args.iter_) 
    #test_accuracy=0 
    initial_energy = model.calculate_energy()
    with open(energy_file,"a") as f: 
            f.write(str(initial_energy))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())
    best_val = 0 
    best_ridge=0
    #test_mlp=0
    iter_ = args.iter_     
    for i in range(num_loop): 
        #start = time.perf_counter() 
        if args.hops == 1 : 
            if args.no_parallel==1:
                energy = model.simulate_2hops_no_parallel(iter_*data.num_nodes) #*data.num_nodes)
            else: 
                energy = model.simulate_2hops(iter_*data.num_nodes)
        else: 

            if args.no_parallel==1: 
                energy = model.simulate_no_parallel(iter_*data.num_nodes)
            else: 
                energy = model.simulate(iter_*data.num_nodes,initial_energy) 
          
        _,emb = model.save_to_numpy(save_file=False)
        initial_energy=energy
        #end=time.perf_counter() 
   
        #total_time = end-start 
        #with open(file_with_time,"a") as f:
         #   f.write(str(total_time))
         #   f.write("\n")
         #   f.flush()
         #   os.fsync(f.fileno())
 
        

        if data_ in ["Photo","Computers","CS","Physics","WikiCS"]: 
            split_file = data_+".npz"
            print("split file =",split_file)
            name_file = os.path.join("data_split",split_file)
            ridge_val, ridge_test = standard_regression(dataset,data,emb,dimension,name_file)
            mlp_val_accuracy,mlp_test_accuracy= mlp_test(dataset,data,emb,dimension,name_file)
        else: 
            ridge_val, ridge_test,std_val,std_test= simple_regression(dataset,data,emb,dimension)   
            
       # iter_=1 
        
       # if mlp_val_accuracy>best_val: 
           # best_val= mlp_val_accuracy
        #name_embedding = "emb_"+str(i)+".npy"
        np.save(os.path.join(tot_dir,"embedding.npy"),emb)  #salva sempre l ultimo embedding e poi sovrascrive 
       # if ridge_val > best_ridge: 
        #    np.save(os.path.join(tot_dir,"emb_ridge.npy"),emb) 
         #   best_ridge=ridge_val   
        with open(file_with_acc,"a") as f: 
            f.write(str(ridge_val))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())
        with open(file_ridge_test,"a") as f:
            f.write(str(ridge_test))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())


        with open(file_with_mlP_acc,"a") as f: 
            f.write(str(mlp_val_accuracy))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())
        with open(file_with_test_mlp,"a") as f: 
            f.write(str(mlp_test_accuracy))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())
        with open(energy_file,"a") as f: 
            f.write(str(initial_energy))
            f.write("\n")
            f.flush() 
            os.fsync(f.fileno())  
        with open(file_with_val_std,"a") as f:
            f.write(str(std_val))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        with open(file_with_test_std,"a") as f:
            f.write(str(std_test))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())


              
           
 #std_val,std_test           
            
            
 # file_with_val_std = os.path.join(tot_dir,"std_val.txt")
  #  file_with_test_std = os.path.join(tot_dir,"std_test.txt")


        

    

    
    
    
    
    
                            
 
                   
                    
                
                    
                        
if __name__ == "__main__": 
    parser = ArgumentParser() 
    parser.add_argument("--dataset", default= 'chameleon') 
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument("--w_connections",help="number of connections in the reservoir",type=int,default=100)
    parser.add_argument("--random_state_w",help="seed of reservoir initialization", type=int,default=5)
    parser.add_argument("--random_state_iteration",type=int,default=5)
    parser.add_argument("--dimension",help="embedding dimension",type=int,default=4096)
    parser.add_argument("--w_type",help="reservoir type, 3 for symmetric", type=int,default=3)
    parser.add_argument("--n_threads",type=int,default=1)
   # parser.add_argument("--run_readout",type=int,default=1) 
    parser.add_argument("--continue_",type=int,default=0) 
    parser.add_argument("--no_parallel",type=int,default=0) 
    parser.add_argument("--bias",help="bias copupling",type=float,default=0)
    parser.add_argument("--hops",help="o if we want aggregate onl the first neighbors, 1 otherwise", type=int,default=0) #lasciare cosi come è 
    parser.add_argument("--iter_",type=int,default=20) #numero di sweeps da fare prima di calcolare accuracy etc    
    parser.add_argument("--threshold",type=float,default=2)  
    parser.add_argument("--temperature",type=float,default=0)
    args =parser.parse_args() 
    main(args)             
            
        
            
    
    
    
    
    
    
    
    
        
        

                    
                    
        
            
    

    
