#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
from torch import optim

import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Polypeptide import three_to_one
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm.notebook import tqdm
import parmap
import time
import os
import warnings
warnings.filterwarnings(action='ignore')


# In[16]:


paser = argparse.ArgumentParser()
args = paser.parse_args("")
args.seed = 123
args.test_size = 0.2
args.shuffle = True


train_pdb_path = './Data/pdb/'
train_graph_path = './Data/graph/'

'''
if you want use custom pdb, put the pdb file in folder '.Custom/pdb'. 
The graph will be generated using pdb file.
'''

test_pdb_path = './Custom/pdb/'
test_graph_path = './Custom/graph/'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[12]:


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
def aa_features(x):
    return np.array(one_of_k_encoding(x, AA))

def adjacency2edgeindex(adjacency):
    start = []
    end = []
    adjacency = adjacency - np.eye(adjacency.shape[0], dtype=int)
    for x in range(adjacency.shape[1]):
        for y in range(adjacency.shape[0]):
            if adjacency[x, y] == 1:
                start.append(x)
                end.append(y)

    edge_index = np.asarray([start, end])
    return edge_index

AMINOS =  ['CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 
           'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET']
def filter_20_amino_acids(array):
    return ( np.in1d(array.res_name, AMINOS) & (array.res_id != -1) )

def protein_analysis_train(pdb_id):
    file_name = rcsb.fetch(pdb_id, "mmtf", train_pdb_path)
    array = strucio.load_structure(file_name)
    protein_mask = filter_20_amino_acids(array)
    try:
        array = array[protein_mask]
    except:
        array = array[0]
        array = array[protein_mask]
    try:
        ca = array[array.atom_name == "CA"]
    except:
        array = array[0]
        ca = array[array.atom_name == "CA"]
    seq = ''.join([three_to_one(str(i).split(' CA')[0][-3:]) for i in ca])
    threshold = 7
    cell_list = struc.CellList(ca, cell_size=threshold)
    A = cell_list.create_adjacency_matrix(threshold)
    A = np.where(A == True, 1, A)
    return [aa_features(aa) for aa in seq], adjacency2edgeindex(A)


def protein_analysis_test(pdb_id):
    array = strucio.load_structure('./Custom/pdb/'+pdb_id+".pdb")
    protein_mask = filter_20_amino_acids(array)
    try:
        array = array[protein_mask]
    except:
        array = array[0]
        array = array[protein_mask]
    try:
        ca = array[array.atom_name == "CA"]
    except:
        array = array[0]
        ca = array[array.atom_name == "CA"]
    seq = ''.join([three_to_one(str(i).split(' CA')[0][-3:]) for i in ca])
    threshold = 7
    cell_list = struc.CellList(ca, cell_size=threshold)
    A = cell_list.create_adjacency_matrix(threshold)
    A = np.where(A == True, 1, A)
    return [aa_features(aa) for aa in seq], adjacency2edgeindex(A)

def pro2vec_train(pdb_id):
    node_f, edge_index = protein_analysis_train(pdb_id)
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))
    return data

def pro2vec_test(pdb_id):
    node_f, edge_index = protein_analysis_test(pdb_id)
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))
    return data


def make_pro(df, target):
    pro_key = []
    pro_value = []
    for i in range(df.shape[0]):
        pro_key.append(df['PDB'].iloc[i])
        pro_value.append(df[target].iloc[i])
    return pro_key, pro_value

def save_graph_train(graph_path, pdb_id):
    vec = pro2vec_train(pdb_id)
    np.save(graph_path+pdb_id+'_e.npy', vec.edge_index)
    np.save(graph_path+pdb_id+'_n.npy', vec.x)

    
def save_graph_test(graph_path, pdb_id):
    vec = pro2vec_test(pdb_id)
    np.save(graph_path+pdb_id+'_e.npy', vec.edge_index)
    np.save(graph_path+pdb_id+'_n.npy', vec.x)
    
def load_graph(graph_path, pdb_id):
    n = np.load(graph_path+pdb_id+'_n.npy')
    e = np.load(graph_path+pdb_id+'_e.npy')
    N = torch.tensor(n, dtype=torch.float)
    E = torch.tensor(e, dtype=torch.long)
    data = Data(x=N, edge_index=E)
    return data

def make_vec(pro, value, graph_path):
    X = []
    Y = []
    mot = []
    for i in range(len(pro)):
        m = pro[i]
        y = value[i]

        v = load_graph(graph_path, m)
        if v.x.shape[0] < 100000:
            X.append(v)
            Y.append(y)
            
    for i, data in enumerate(X):
        y = Y[i]
        #data.y = torch.tensor([y], dtype=torch.long)
        data.y = torch.tensor([y], dtype=torch.float)#flaot
    return X

def generate_graph_train(pdb, graph_path):
    done = 0
    while done == 0:
        graph_dirs = list(set([d[:-6] for d in os.listdir(graph_path)]))
        if pdb not in graph_dirs:
            try:
                save_graph_train(graph_path,pdb)
                done = 1
                return 1
            except:
                done = 1
                return 0
        else:
            done = 1
            return 1
        
        
def generate_graph_test(pdb, graph_path):
    done = 0
    graph_dirs = list(set([d[:-6] for d in os.listdir(graph_path)]))
    while done == 0:
        if pdb not in graph_dirs:
            try:
                save_graph_test(graph_path,pdb)
                done = 1
                return 1
            except:
                done = 1
                return 0
        else:
            done = 1
            return 1        


# In[3]:


def save_checkpoint(epoch, model, optimizer, filename):
    state = {'Epoch': epoch, 
             'State_dict': model.state_dict(), 
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)

def train(model, device, optimizer, train_loader, criterion, args):
    epoch_train_loss = 0
    for i, pro in enumerate(train_loader):
        pro, labels= pro.to(device), pro.y.to(device)
        optimizer.zero_grad()
        outputs = model(pro)
        outputs.require_grad = False
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs.flatten(), labels)
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_train_loss /= len(train_loader)
    return model, epoch_train_loss

def test(model, device, test_loader, criterion, args):
    model.eval()
    data_total = []
    pred_data_total = []
    epoch_test_loss = 0
    with torch.no_grad():
        for i, pro in enumerate(test_loader):
            pro, labels= pro.to(device), pro.y.to(device)
            data_total += pro.y.tolist()
            outputs = model(pro)
            pred_data_total += outputs.view(-1).tolist()
            loss = criterion(outputs.flatten(), labels)
            epoch_test_loss += loss.item()
    epoch_test_loss /= len(test_loader)
    return data_total, pred_data_total, epoch_test_loss

def experiment(model, train_loader, test_loader, device, args):
    time_start = time.time()
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criterion = nn.MSELoss() #L2 loss 
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)
    
    list_train_loss = []
    list_test_loss = []
    print('[Train]')
    for epoch in range(args.epoch):
        scheduler.step()
        model, train_loss = train(model, device, optimizer, train_loader, criterion, args)
        _, _, test_loss = test(model, device, test_loader, criterion, args)
        list_train_loss.append(train_loss)
        list_test_loss.append(test_loss)
        print('- Epoch : {0}, Train Loss : {1:0.4f}, Test Loss : {2:0.4f}'.
              format(epoch+1, train_loss, test_loss))
    
    print()
    print('[Test]')
    data_total, pred_data_total, _ = test(model, device, test_loader, criterion, args)
    print('- R2 : {0:0.4f}'.format(r2_score(data_total, pred_data_total)))
    
    print(type(data_total), type(pred_data_total))
    solution=pd.DataFrame(data_total, columns=["test"])
    answer=pd.DataFrame(pred_data_total, columns=["answer"])
    csv=pd.concat([solution, answer],axis=1)
    args.csv = csv
    time_end = time.time()
    time_required = time_end - time_start
    
    args.list_train_loss = list_train_loss
    args.list_test_loss = list_test_loss
    args.data_total = data_total
    args.pred_data_total = pred_data_total
    
    save_checkpoint(epoch, model, optimizer, './mymodel.pt')
    
    return args


# In[4]:


class GCNlayer(nn.Module):
    def __init__(self, n_features, conv_dim1, conv_dim2, conv_dim3, conv_dim4, concat_dim, dropout):
        super(GCNlayer, self).__init__()
        self.n_features = n_features
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.conv_dim4 = conv_dim4
        self.concat_dim =  concat_dim
        self.dropout = dropout
        
        self.conv1 = GCNConv(self.n_features, self.conv_dim1)
        self.bn1 = BatchNorm1d(self.conv_dim1)
        self.conv2 = GCNConv(self.conv_dim1, self.conv_dim2)
        self.bn2 = BatchNorm1d(self.conv_dim2)
        self.conv3 = GCNConv(self.conv_dim2, self.conv_dim3)
        self.bn3 = BatchNorm1d(self.conv_dim3)
        self.conv4 = GCNConv(self.conv_dim3, self.conv_dim4)
        self.bn4 = BatchNorm1d(self.conv_dim4)
        self.conv5 = GCNConv(self.conv_dim4, self.concat_dim)
        self.bn5 = BatchNorm1d(self.concat_dim)
        
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class FClayer(nn.Module):
    def __init__(self, concat_dim, pred_dim1, pred_dim2, pred_dim3, pred_dim4, out_dim, dropout):
        super(FClayer, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1
        self.pred_dim2 = pred_dim2
        self.pred_dim3 = pred_dim3
        self.pred_dim4 = pred_dim4        
        self.out_dim = out_dim
        self.dropout = dropout

        self.fc1 = Linear(self.concat_dim, self.pred_dim1)
        self.fc2 = Linear(self.pred_dim1, self.pred_dim2)
        self.fc3 = Linear(self.pred_dim2, self.pred_dim3)
        self.fc4 = Linear(self.pred_dim3, self.pred_dim4)
        self.fc5 = Linear(self.pred_dim4, self.out_dim)
    
    def forward(self, data):
        x = self.fc1(data)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x
    
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = GCNlayer(args.n_features, 
                              args.conv_dim1, 
                              args.conv_dim2, 
                              args.conv_dim3,
                              args.conv_dim4,
                              args.concat_dim, 
                              args.dropout)
        
        self.fc = FClayer(args.concat_dim, 
                          args.pred_dim1, 
                          args.pred_dim2, 
                          args.pred_dim3,
                          args.pred_dim4,
                          args.out_dim, 
                          args.dropout)
        
    def forward(self, pro):
        x = self.conv1(pro)
        x = self.fc(x)
        return x


# In[19]:


df = pd.read_csv('Inner3.csv')
df = df[0:100]
df


# In[20]:


check = parmap.map(generate_graph_train, [pdb for pdb in df['PDB'].tolist()], "./Data/graph/", pm_pbar=True, pm_processes=200)
df['Check'] = check
df = df[df['Check'] == 1].reset_index(drop=True)
df


# In[21]:


#if you want to Custom data, please revise this code.

test_df = pd.read_csv('Custom_data.csv')
test_df

check = parmap.map(generate_graph_test, [pdb for pdb in test_df['PDB'].tolist()], "./Custom/graph/", pm_pbar=True, pm_processes=200)
test_df['Check'] = check
test_df = test_df[test_df['Check'] == 1].reset_index(drop=True)
print(test_df)


# In[22]:


X_train, X_val= train_test_split(df, test_size=0.2, random_state=1)
X_test = test_df

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

train_pro_key, train_pro_value = make_pro(X_train, "y_static")
val_pro_key, val_pro_value = make_pro(X_val, "y_static")
test_pro_key, test_pro_value = make_pro(X_test, "y_static")
train_X = make_vec(train_pro_key, train_pro_value, train_graph_path)
val_X = make_vec(val_pro_key, val_pro_value, train_graph_path)
test_X = make_vec(test_pro_key, test_pro_value, test_graph_path)


# In[24]:


prediction=pd.DataFrame()
prediction["PDB"]=test_df["PDB"]
prediction["y_static"]=test_df["y_static"]

args.batch_size = 50
args.epoch = 50
args.lr = 0.001
args.optim = 'Adam'
args.step_size = 10
args.gamma = 0.9
args.dropout = 0.2
args.n_features = 20
dim = 8
args.conv_dim1 = dim
args.conv_dim2 = 2*dim
args.conv_dim3 = 4*dim
args.conv_dim4 = 2*dim
args.concat_dim = dim
args.pred_dim1 = dim
args.pred_dim2 = 2*dim
args.pred_dim3 = 4*dim
args.pred_dim4 = 2*dim
args.out_dim = 1

model = Net(args)
model = model.to(device)

train_loader = DataLoader(train_X, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_X, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_X, batch_size=args.batch_size, shuffle=False, drop_last=False)


dict_result = dict()
args.exp_name = 'Protein Aggragation'
result = vars(experiment(model, train_loader, val_loader, device, args))
dict_result[args.exp_name] = result
torch.cuda.empty_cache()

model.eval()
data_total = []
pred_data_total = []
epoch_test_loss = 0
criterion = nn.MSELoss()

with torch.no_grad():
    for i, pro in enumerate(test_loader):
        pro, labels= pro.to(device), pro.y.to(device)
        data_total += pro.y.tolist()
        outputs = model(pro)
        pred_data_total += outputs.view(-1).tolist()
        loss = criterion(outputs.flatten(), labels)
        epoch_test_loss += loss.item()
epoch_test_loss /= len(val_loader)
print(r2_score(data_total, pred_data_total))
pred_data_total = pd.DataFrame(pred_data_total, columns=["prediction"]) 
prediction=pd.concat([prediction, pred_data_total], axis=1)
prediction.to_csv("Regression_output.csv")

