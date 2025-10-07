import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prompts.AllinonePrompt import HeavyPrompt
from prompts.Gprompt import Gprompt
from prompts.GPPTPrompt import GPPTPrompt 
from prompts.GPFPrompt import GPF,GPF_plus
from ProG.model.GCN import GCN as ProGGCN
from surrogate_models.GCN import GCN
from surrogate_models.GRACE import GRACE,trainGRACE,trainClass,Encoder,clr_loss,bkd_loss
from surrogate_models.Homo import Homo
import numpy as np
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from clustering_nodes.cluster_graph import cluster_subgraph
from surrogate_models.GraphEncoder import GCN_Encoder
from surrogate_models.prompt_select import PromptSelector
from utils import calculate_accuracy,calculate_similarity,get_similar_prompt,find_k_subgraphs_per_node,adj_to_edges_batch,adj_to_edges
from tqdm import tqdm
from tqdm.contrib import tzip
from surrogate_models.GraphEncoder import initialize_weights 
from torch_geometric.utils import k_hop_subgraph
import random
import logging
from utils import prune_unrelated_edge,count_removed_edges,calculate_weights,reconstruct_prune_unrelated_edge,label_num,random_select
from utils import check_model_params_unchanged,random_sample_per_class,train_classifer_without_val
from construct import model_construct
import copy     
import os 
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None
          
class PromptPool(torch.nn.Module):
    def __init__(self,args,device,feature,idx_attach,edge_index,labels):
        super(PromptPool, self).__init__()
        self.device = device
        self.edge_index = edge_index
        self.labels = labels
        self.args = args
        self.target_node = idx_attach
        self.num_prompts = int(len(idx_attach))
        self.feature = feature.to(self.device)
        self.shared_edge_weights = None
        self.shared_prompt_pool = None
        self.promptPool = None
        self.promptEdges = None
        layers = []
        dropout = 0.00
        layernum = 1
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(feature.shape[1], feature.shape[1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)
        if args.mode:
            self.feat = nn.Linear(feature.shape[1],args.prompt_size*feature.shape[1]).to(device)
        else:
            self.feat = nn.Linear(feature.shape[1],feature.shape[1]).to(device)
        self.edge = nn.Linear(feature.shape[1], int(args.prompt_size*(args.prompt_size-1)/2)).to(device)
        
        
    def init_subgraph(self):
        subgraphs = find_k_subgraphs_per_node(self.args,self.edge_index, range(self.feature.shape[0]), self.args.prompt_size,self.args.max_subgraph  )
        return subgraphs
    
    def init_promptPool(self,encoder):
        subgraphs = self.init_subgraph()
        subgraph_edges = []
        subgraph_nodes = []
        for sg in subgraphs:
            subgraph_nodes.append(sg.nodes())
            subgraph_edges.append(sg.edges())
        promptpool = []
        _,edges,index_list = cluster_subgraph(encoder,self.feature,subgraph_edges,subgraph_nodes,self.args.num_prompts,self.args,self.device)
        
        # prompt 
        for idx in index_list:
            prompt = self.feature[list(subgraphs[idx].nodes())]
            promptpool.append(prompt)
        
        promptpool = torch.cat(promptpool,dim=0).reshape(self.args.num_prompts,self.args.prompt_size,-1).to(self.device)
        # selected_prompts = []
        # for i in range(self.args.num_prompts):
        #     random_index = torch.randint(0, promptpool.shape[1], (1,))[0].item()
        #     selected_prompt = promptpool[i, random_index, :]
        #     selected_prompts.append(selected_prompt)
        # self.selected_prompts = torch.stack(selected_prompts)# 
    
        self.promptPool =nn.Parameter(promptpool.clone().detach()) 

        # prompt 
        self.promptEdges = edges
        
        
    def update(self):
        pool,edge_weight = self.forward()
        self.shared_edge_weights = edge_weight
        self.shared_prompt_pool = pool
        
    def forward(self):
        GW = GradWhere.apply
        self.layers = self.layers
        origin = self.promptPool.clone()
        h = self.layers(origin.reshape(-1,self.feature.shape[1]))
        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, 0.5, self.device)
       
        return self.promptPool,edge_weight
    

from torch_geometric.data import Data, Batch
import itertools
from ProG.utils.constraint import constraint
#unnoticeable injection atttack prompt
from utils import merge_graphs,getTargetCLass,prune_unrelated_edge_isolated,getRandomClass,getSelfClass,train_classifer,train_classifer_cora
class CPGBA(torch.nn.Module):
    def __init__(self,args, device,data,idx_attach):
        super(CPGBA, self).__init__() 
        self.args = args
        self.device = device
        self.weights = None
        self.data = data
        self.homo = Homo(data.x.shape[1],args.hidden,device)
        self.prompt = PromptPool(args,device,data.x.clone(),idx_attach,data.edge_index.clone(),data.y.clone()).to(device)#center features
        self.idx_attach = idx_attach
    #  
    def get_trigger_index(self):
        
        edge_list = []
        edge_list.append([0,0])
        for j in range(self.args.prompt_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index
    
    
    def get_inject_edge(self,start,injected_features, idx_attach,mode,edge=None):
        
        idx_feature = self.features[idx_attach]  # 
        similarities = F.cosine_similarity(idx_feature, injected_features, dim=1)
        index = similarities.argmax()
        
        if mode=='N':
            edge_list = []
            edges = self.get_trigger_index().clone()
            edges[0,0] = idx_attach
            edges[1,0] = start+index
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            
            edge_index = torch.cat(edge_list,dim=1)
            row = torch.cat([edge_index[0], edge_index[1]])
            col = torch.cat([edge_index[1],edge_index[0]])
            edge_index = torch.stack([row,col])
            connect_num = 1
        elif mode=='Y':
            sorted_edges = torch.sort(edge, dim=0)[0]
            sorted_edges = sorted_edges.t()
            unique_edges = torch.unique(sorted_edges, dim=0)
            origin_edge = unique_edges.t()
            
            new_edge = torch.tensor([
                [idx_attach],
                [start + index]
            ]).to(self.device)
            adjusted_edge_index = origin_edge + start
            edge_index = torch.cat((new_edge, adjusted_edge_index), dim=1)
            row = torch.cat([edge_index[0], edge_index[1]])
            col = torch.cat([edge_index[1],edge_index[0]])
            edge_index = torch.stack([row,col])
            connect_num = 1
        elif mode=='F':
            sorted_edges = torch.sort(edge, dim=0)[0]
            sorted_edges = sorted_edges.t()
            unique_edges = torch.unique(sorted_edges, dim=0)
            origin_edge = unique_edges.t()
            edge_list = []  
            
            
            high_similarity_indices = torch.where(similarities > self.args.test_thr)[0] 
            connect_num = high_similarity_indices.size(0)
            if high_similarity_indices.numel() == 0: 
                high_similarity_indices  = torch.tensor([index])
                connect_num = 1
            
            
            for i in high_similarity_indices:  
              
                target_index = start + i  
                edge = torch.tensor([[idx_attach], [target_index]], dtype=torch.long).to(self.device)  
                edge_list.append(edge)  
            
            
            new_edge = torch.cat(edge_list, dim=1).to(self.device)
        
            adjusted_edge_index = origin_edge + start
            edge_index = torch.cat((new_edge, adjusted_edge_index), dim=1)
            row = torch.cat([edge_index[0], edge_index[1]])
            col = torch.cat([edge_index[1],edge_index[0]])
            edge_index = torch.stack([row,col])
        return edge_index,connect_num
    
    def getFineTune(self,trojan_feat):
        if self.args.finetune:
            prompts_x = self.homo(trojan_feat)
          
            return prompts_x 
        else:
            return trojan_feat
         
    
    def get_trojan(self,edges_pool,prompt_pool,weights_pool,idx_attach,features,total_edge_index,target_classes,mode='F'):
        trojan_feat_list = []
        trojan_weight_list = []
        trojan_edge_list = []
        start = len(features)
        origin_feat = []
        if self.args.exp==6 or self.args.exp==7:#
            attack_embedding = []
            
            
           
        if self.args.position:
            embed_pool = self.getFineTune(prompt_pool)
        for index,idx in enumerate(idx_attach):
            if self.args.exp==4 or self.args.exp==5:
                idx_prompt= get_similar_prompt(self.device,features[idx,:],self.args.prompt_size,prompt_pool,None,None,None,None)
            elif self.args.exp==6:
                # pass
                idx_prompt= get_similar_prompt(self.device,features[idx,:],self.args.prompt_size,embed_pool,self.label_full,self.label_origin,target_classes[index],mode)
                edge = edges_pool[idx_prompt].to(self.device)
            elif self.args.exp==7:
               
                features = features.to(self.device)
                prompt_pool = prompt_pool.to(self.device)
                
                if self.args.position:
                    
                    idx_prompt= get_similar_prompt(self.device,features[idx,:],self.args.prompt_size,embed_pool,self.label_full,self.label_origin,target_classes[index],mode)
                else:
                    idx_prompt= get_similar_prompt(self.device,features[idx,:],self.args.prompt_size,prompt_pool,self.label_full,self.label_origin,target_classes[index],mode)
                # idx_yes_s= get_similar_prompt(self.device,attack_nodes_embedding[index],self.args.prompt_size,embed_pool,self.label_full,self.label_origin,target_classes[index],'O')
                # edge = new_edges_pool[idx_yes_s]
                # s_prompt = prompt_pool[idx_yes_s]
                # s_weight = weights_pool[idx_yes_s]
            elif self.args.exp==8:
                idx_prompt= get_similar_prompt(self.device,features[idx,:],self.args.prompt_size,embed_pool,self.label_full,self.label_origin,target_classes[index],mode)
                edge = edges_pool[idx_prompt].to(self.device)
          
            if self.args.position:
                prompt = embed_pool[idx_prompt]
                origin_feat.append(prompt_pool[idx_prompt])
            else:
                prompt = prompt_pool[idx_prompt]
            weight = weights_pool[idx_prompt]
            
            trojan_feat_list.append(prompt)
            if self.args.exp==7:
                pass
                # trojan_feat_list.append(s_prompt)
                
            if self.args.exp==4 or self.args.exp==5:
                insert_edge_index,connect_num = self.get_inject_edge(start,prompt,idx,'N')
                insert_edge_index = insert_edge_index.to(self.device)
                trojan_edge_list.append(insert_edge_index)# 
                
            elif self.args.exp==6:
                # pass
                edge = adj_to_edges(edge).to(self.device)
                insert_edge_index,connect_num =self.get_inject_edge(start,prompt,idx,'Y',edge=edge)
                insert_edge_index = insert_edge_index.to(self.device)
                trojan_edge_list.append(insert_edge_index)# 
                
            elif self.args.exp==7:
                insert_edge_index,connect_num = self.get_inject_edge(start,prompt,idx,'N')
                trojan_edge_list.append(insert_edge_index)# 
            elif self.args.exp==8: 
                edge = adj_to_edges(edge).to(self.device)
                insert_edge_index,connect_num = self.get_inject_edge(start,prompt,idx,'F',edge=edge)
                insert_edge_index = insert_edge_index.to(self.device)
                trojan_edge_list.append(insert_edge_index)
            
            start+=self.args.prompt_size
            if self.args.exp==7:
                pass
                # trojan_edge_list.append(self.get_inject_edge(start,s_prompt,idx,'Y',edge=edge).to(self.device))# 
                # start+=self.args.prompt_size
            if self.args.exp==4 or self.args.exp==5 or self.args.exp==6:
                weight = weight[:int(trojan_edge_list[-1].shape[1]/2)-1]
            elif self.args.exp==7:
                weight = weight[:int(trojan_edge_list[-1].shape[1]/2)-1]
            elif self.args.exp==8:
                
                weight = weight[:int(trojan_edge_list[-1].shape[1]/2)-connect_num]
                # print(weight.shape,s_weight[:int(trojan_edge_list[-1].shape[1]/2)-1].shape)
                # weight = torch.cat((weight, s_weight[:int(trojan_edge_list[-1].shape[1]/2)]), dim=0)

            if self.args.exp==7 or self.args.exp==4 or self.args.exp==5 or self.args.exp==6:
                trojan_weight_list.append(torch.cat([torch.ones([1],dtype=torch.float,device=self.device),weight]))
            elif self.args.exp==8:
                trojan_weight_list.append(torch.cat([torch.ones(connect_num,dtype=torch.float,device=self.device),weight]))
            
        if len(trojan_feat_list) == 0:
            trojan_feat = None
            trojan_weights = None
            trojan_edge = None
        else:
            trojan_feat = torch.cat(trojan_feat_list,dim=0)
            trojan_weights = torch.cat(trojan_weight_list,dim=0)
            trojan_edge = torch.cat(trojan_edge_list,dim=1)
        
        regularization_loss = None
        if len(trojan_feat_list) != 0:
            if self.args.position:
                origin_feat = torch.cat(trojan_feat_list,dim=0)
                regularization_loss = torch.norm(trojan_feat - origin_feat, p=2)
        return trojan_feat, trojan_weights,trojan_edge,regularization_loss


    def get_attach_poision(self, idx_attach, features,edge_index,edge_weight,device,target):
        self.prompt = self.prompt.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.prompt.eval()
        self.homo.eval()
        # self.newEncoder.eval()
        if target is not None:
            target_classes = getTargetCLass(idx_attach,self.label_full,self.label_origin,self.orginLabels,target).to(self.device)
        else:
            target_classes = getTargetCLass(idx_attach,self.label_full,self.label_origin,self.orginLabels).to(self.device)
        trojan_feat, trojan_weights,trojan_edge,_ = self.get_trojan(self.prompt.promptEdges,self.prompt.shared_prompt_pool,self.prompt.shared_edge_weights,idx_attach,features,edge_index,target_classes = target_classes)
        
        trojan_edge = trojan_edge.reshape(edge_index.shape[0],-1)
        if self.args.position:
            pass
        else:
            trojan_feat = self.getFineTune(trojan_feat)
       
        poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]) # repeat trojan weights beacuse of undirected edge
        poison_x = torch.cat([features,trojan_feat])
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        
        return poison_x, poison_edge_index, poison_edge_weights,trojan_feat,trojan_edge,target_classes
    
    def get_random_attach_poision(self, idx_attach, labels,features,edge_index,edge_weight,device):
        self.prompt = self.prompt.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.prompt.eval()
        self.homo.eval()
        # self.newEncoder.eval()
        
        if self.args.attack_single:           
            target_classes = torch.zeros(len(idx_attach), dtype=torch.long).to(self.device)
            target_classes = target_classes.fill_(self.args.target_label)
        else:
            target_classes = getRandomClass(idx_attach,self.args.attack_num).to(self.device)
        # target_classes = getRandomClass(idx_attach,self.args.attack_num).to(self.device)
        # target_classes = getRandomClass(idx_attach,labels.max().item()+1).to(self.device)
        trojan_feat, trojan_weights,trojan_edge,_ = self.get_trojan(self.prompt.promptEdges,self.prompt.shared_prompt_pool,self.prompt.shared_edge_weights,idx_attach,features,edge_index,target_classes = target_classes)
        
        if trojan_feat == None:
            poison_edge_weights = edge_weight# repeat trojan weights beacuse of undirected edge
            poison_x = features
            poison_edge_index = edge_index
        else:
            trojan_edge = trojan_edge.reshape(edge_index.shape[0],-1)
            if self.args.position:
                pass
            else:
                trojan_feat = self.getFineTune(trojan_feat)
       
            poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]) # repeat trojan weights beacuse of undirected edge
            poison_x = torch.cat([features,trojan_feat])
            poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        
        return poison_x, poison_edge_index, poison_edge_weights,trojan_feat,trojan_edge,target_classes
    
    
    def getPromptLabels(self,model,features,edge_index,labels,idx_train):
        count_labels2 = [0]* (labels.max().item()+1)
        count_labels1 = [0]* (labels.max().item()+1)
        max_count = self.args.prompt_size/(labels.max().item()+1)
        subgraph_edges = []
        subgraph_nodes = []
        center_nodes = []
        subgraph_features = []
        for _ in range(100):
            center_node = random.choice(range(features.shape[0]))
            try:
                subgraph_node_idx, subgraph_edge_index, sub_mapping, _  = k_hop_subgraph(center_node, 2, edge_index, relabel_nodes=True)
            except RuntimeError:
                continue
            
            center_nodes.append(sub_mapping.item())
            subgraph_nodes.append(subgraph_node_idx)
            subgraph_edges.append(subgraph_edge_index)
            subgraph_features.append(features[subgraph_node_idx])
        promptEdges = self.prompt.promptEdges.detach().cpu()
        prompt = self.prompt.promptPool
        node_pairs = list(itertools.combinations(range(len(prompt[0])), 2))
        edges = []
        for u, v in node_pairs:
            edges.append([u, v])
        for u, v in node_pairs:
            edges.append([v, u])
        
        full_connected_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        label1 = []
        label2 = []
        for idx in range(len(prompt)):
            diff_sum1 = torch.zeros(labels.max().item()+1, device=self.device)
            diff_sum2 = torch.zeros(labels.max().item()+1, device=self.device)
            attack_full_connected_graph = Data(x=prompt[idx], edge_index=full_connected_edge_index).to(self.device)
            attack_origin_connected_graph = Data(x=prompt[idx], edge_index=adj_to_edges(promptEdges[idx])).to(self.device)
            
            for i in range(100):
                orgin_graph = Data(x=subgraph_features[i], edge_index=subgraph_edges[i]).to(self.device)
                
                attack1_grapgh = merge_graphs(self.device,orgin_graph, attack_full_connected_graph, center_nodes[i])
                attack2_grapgh = merge_graphs(self.device,orgin_graph, attack_origin_connected_graph, center_nodes[i])
                origin_output = model(orgin_graph.x, orgin_graph.edge_index)
                attack_full_output = model(attack1_grapgh.x, attack1_grapgh.edge_index)
                attack_origin_output =  model(attack2_grapgh.x, attack2_grapgh.edge_index)
                diff_sum1 += (attack_full_output[center_nodes[i]] - origin_output[center_nodes[i]])
                diff_sum2 += (attack_origin_output[center_nodes[i]] - origin_output[center_nodes[i]])
            diff_avg1 = diff_sum1 / 100
            diff_avg2 = diff_sum2 / 100
            argmax_label1 = torch.argmax(diff_avg1).item()
            argmax_label2 = torch.argmax(diff_avg2).item()
            
            if count_labels1[argmax_label1] >= max_count:
                # array = list(range(0, labels.max().item()))
                random_index = random.randint(0, labels.max().item())
                label1.append(random_index)
            else:
                count_labels1[argmax_label1]+=1
                label1.append(argmax_label1)
            if count_labels2[argmax_label2] >= max_count:
                random_index = random.randint(0, labels.max().item())
                label2.append(random_index)
            else:
                count_labels2[argmax_label2] +=1
                label2.append(torch.argmax(diff_avg2).item())
            
        return label1,label2
    
    
    def fit(self,features, edge_index, edge_weight, labels, idx_train, idx_val,idx_attach,idx_unlabeled,idx_atk,idx_clean_test,mask_edge_index):
        ori_features = features.clone()
        self.idx_train = idx_train
        rs = np.random.RandomState(self.args.seed)
        idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=512,replace=False)]])
       
        if self.args.train == False:
            print('==test begin==',flush=True)
            file_path = './parameters/{}_{}.pth'.format(self.args.dataset,self.args.index)
            total_weights = torch.load(file_path)  
            weights = total_weights['weights']
            homo_weights = total_weights['homo_weights']
            label_full = total_weights['label_full']
            label_origin = total_weights['label_origin']
            orginLabels = total_weights['orginLabels']
            val_idx_attach = total_weights['val_idx_attach']
            idx_attach = val_idx_attach
            

        log_file = 'log/{}_record.log'.format(self.args.dataset)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info('args:{}'.format(self.args))
        
    
        self.orginLabels = labels.clone()
        args = self.args
        device = self.device
        modes = ['F','R']
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=device,dtype=torch.float)
        self.idx_attach = idx_attach
        
        from surrogate_models.MLP import MLP
        projection = MLP(features.shape[1], 1433, labels.max().item() + 1, dropout=0.5, lr=args.train_lr, weight_decay=args.weight_decay, device=device).to(device)
        projection.fit(features, labels, idx_train, idx_val, train_iters=1000, initialize=True, verbose=False)
        acc = projection.test(idx_clean_test)
        print('projection-ACC:{:.4f}'.format(acc),flush=True)
        with torch.no_grad():
            features = projection.get_h(features)
        self.features = features
        self.features = self.features.to(device)
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        self.selector = PromptSelector(args,device,self.features.shape[1])
        
        
        self.encoder = GCN_Encoder(self.features.shape[1], self.args.hidden,labels.max().item()+1,layer=1,device=device).to(device)
        print('=== training encoder model ===',flush=True)
        self.encoder.fit(self.features, self.edge_index, None, labels, idx_train, idx_val, train_iters=200, verbose=True)
        print('=== end  ===',flush=True)
        torch.save(self.encoder.state_dict(), './original_model_{}.pth'.format(args.dataset))
      
        
        
        file_path = './clean-{}-gcl.pth'.format(args.dataset)
     
        # encoder = Encoder(features.shape[1], args.hidden,k=args.layer).to(device)
        # clean_model = GRACE(encoder = encoder,
        #                 nfeat=features.shape[1],
        #                 nhid=args.hidden,
        #                 nclass=labels.max().item() + 1,
        #                 layer = args.layer,
        #                 dropout=0.5, device=self.device,args = self.args).to(device)
        # if os.path.exists(file_path):
        #     clean_model.load_state_dict(torch.load(file_path))
        # else:
        #     pass
        # self.surrogate_model = copy.deepcopy(clean_model)
        print('proj dim:',features.shape[1],flush=True)
        self.surrogate_model = ProGGCN(input_dim=features.shape[1], out_dim=args.hidden, num_layer=args.layer)
        self.surrogate_model.load_state_dict(torch.load(args.pre_train_model_path, map_location=device))
        self.surrogate_model = self.surrogate_model.to(device)
        
        
        if args.prompt_type == 'GPPT':
            self.prompt_origin = GPPTPrompt(args.hidden, labels.max().item() + 1, labels.max().item() + 1, device = device)
            node_embedding = self.surrogate_model(features, edge_index)
            self.prompt_origin.weigth_init(node_embedding,edge_index, labels, idx_train)
        elif args.prompt_type == 'GPF':
            self.prompt_origin = GPF(features.shape[1]).to(device)
            self.answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                            torch.nn.Softmax(dim=1)).to(device)
        elif args.prompt_type == 'Gprompt':
            self.prompt_origin = Gprompt(args.hidden).to(device)
            self.answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                            torch.nn.Softmax(dim=1)).to(device)
        elif args.prompt_type == 'All-in-one':
            self.prompt_origin = HeavyPrompt(token_dim=features.shape[1], token_num=3, cross_prune=0.1, inner_prune=0.3).to(device)
            self.answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                            torch.nn.Softmax(dim=1)).to(device)
            
            
        self.loss  = F.cross_entropy
        self.homoss = HomoLoss(args,device)
        self.homo = Homo(features.shape[1],args.hidden,device)
            
        # trigger pool
        self.prompt = PromptPool(args,device,features,idx_attach,edge_index,labels).to(device)
        self.prompt.init_promptPool(self.encoder)
        print("===label pool begin==",flush=True)
        self.label_full,self.label_origin = self.getPromptLabels(self.encoder,features,edge_index,labels,idx_train)
        print("===label pool over==",flush=True)
       
        
        optimizer_selector = optim.Adam(self.selector.parameters(),lr=self.args.train_lr, weight_decay=self.args.weight_decay)
        optimizer_prompt_pool = optim.Adam(self.prompt.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        optimizer_homo = optim.Adam(self.homo.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)  
        if args.prompt_type == 'GPPT':
            optimizer_prompt_origin = optim.Adam(self.prompt_origin.parameters(), lr=args.train_lr, weight_decay=args.weight_decay) 
            print('GPPT',flush=True)
        elif args.prompt_type == 'GPF' or args.prompt_type == 'Gprompt':
            model_param_group = []
            model_param_group.append({"params": self.prompt_origin.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            optimizer_prompt_origin = optim.Adam(model_param_group, lr=args.train_lr, weight_decay=args.weight_decay)
            print(args.prompt_type,flush=True)
        elif args.prompt_type == 'All-in-one':
            optimizer_prompt_origin = optim.Adam(filter(lambda p: p.requires_grad, self.prompt_origin.parameters()), lr=args.train_lr, weight_decay= args.weight_decay)
            optimizer_answer = optim.Adam(filter(lambda p: p.requires_grad, self.answering.parameters()), lr=args.train_lr, weight_decay= args.weight_decay)
            print('All-in-one',flush=True)
        loss_best = -1        
        self.prompt.update()
        # self.target_labels = [0,1,2,3]
        # args.attack_single = True
        if args.train:
            
            for i in range(args.trojan_epochs):
              
                self.prompt.eval()
                self.homo.eval()
                self.prompt_origin.train()
                for j in range(args.inner):
                    optimizer_prompt_origin.zero_grad()
                    if  args.prompt_type == 'All-in-one':
                        optimizer_answer.zero_grad()
                    
                    for item in modes:
                        if args.attack_single:
                            idx_attach_target_classes = torch.zeros(len(idx_attach), dtype=torch.long).to(device)
                            idx_attach_target_classes = idx_attach_target_classes.fill_(args.target_label)
                        else:
                            idx_attach_target_classes = getRandomClass(idx_attach,args.attack_num).to(device)

                        self.labels = labels.clone()
                        self.labels[idx_attach] = idx_attach_target_classes
                        self.homo.eval()
                        
                        if item == 'R':
                            trojan_feat, trojan_weights,trojan_edge,_= self.get_trojan(self.prompt.promptEdges,self.prompt.shared_prompt_pool,self.prompt.shared_edge_weights,idx_attach,features,self.edge_index,target_classes=idx_attach_target_classes,mode='R') # may revise the process of generate
                        elif item == 'F':
                            trojan_feat, trojan_weights,trojan_edge,_= self.get_trojan(self.prompt.promptEdges,self.prompt.shared_prompt_pool,self.prompt.shared_edge_weights,idx_attach,features,self.edge_index,target_classes=idx_attach_target_classes) # may revise the process of generate
                        trojan_edge = trojan_edge.reshape(edge_index.shape[0],-1)
                        
                        if self.args.position:
                            pass
                        else:
                            trojan_feat = self.getFineTune(trojan_feat)
                        
                        
                            
                       
                        poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                        poison_x = torch.cat([features,trojan_feat]).detach()
                        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)
                        
                        if args.prompt_type == 'GPF':
                            poison_x = self.prompt_origin.add(poison_x)
                        elif args.prompt_type == 'All-in-one':
                            pg_data = Data(x=poison_x, edge_index=poison_edge_index)
                            pg_poison_x, pg_poison_edge_index= self.prompt_origin(pg_data, torch.cat([idx_train,idx_attach]))
                            poison_edge_weights = torch.cat([torch.ones([(pg_poison_edge_index.shape[1]-poison_edge_index.shape[1])//2],dtype=torch.float, device=device),poison_edge_weights]).to(device)
                        if args.prompt_type == 'All-in-one':
                            embedding = self.surrogate_model(pg_poison_x, pg_poison_edge_index,poison_edge_weights)
                        else:  
                            embedding = self.surrogate_model(poison_x, poison_edge_index,poison_edge_weights)
                        
                        if args.prompt_type == 'GPF' or args.prompt_type == 'All-in-one':
                            output = self.answering(embedding)
                        elif args.prompt_type == 'GPPT':
                            output = self.prompt_origin(embedding, poison_edge_index)
                        elif args.prompt_type == 'Gprompt':
                            embedding = self.prompt_origin(embedding)
                            # embedding = global_mean_pool(embedding)
                            output = self.answering(embedding)
                           
                        # clean weight/trian 
                        loss_inner = self.loss(output[idx_attach], self.labels[idx_attach])  + args.clr_weight * self.loss(output[idx_train], self.labels[idx_train]) 
                        # loss_inner = self.loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) 
                        # if args.prompt_type == 'GPPT':
                        #     loss_inner += 0.001 * constraint(device, self.prompt_origin.get_TaskToken())
                        loss_inner.backward(retain_graph=True) 
                        
                    
                
                    optimizer_prompt_origin.step()
                    if args.prompt_type == 'All-in-one':
                        optimizer_answer.step()
                    if args.prompt_type == 'GPPT':
                        self.prompt_origin.update_StructureToken_weight(self.prompt_origin.get_mid_h())
                    
                    acc_train = calculate_accuracy(output[idx_train],self.labels[idx_train])
                    acc_train_attach = calculate_accuracy(output[idx_attach],self.labels[idx_attach])
                    
                
                self.prompt.train()
                self.homo.train()
                self.prompt_origin.eval()
                optimizer_prompt_pool.zero_grad()
                optimizer_homo.zero_grad()
                optimizer_selector.zero_grad()
       
                
                for item in modes:
                    if args.attack_single:
                        idx_outter_target_classes = torch.zeros(len(idx_outter), dtype=torch.long).to(device)
                        idx_outter_target_classes = idx_outter_target_classes.fill_(args.target_label)
                    else:
                        idx_outter_target_classes = getRandomClass(idx_outter,args.attack_num).to(device)
        
           
                    if item == 'R':
                        trojan_feat, trojan_weights,trojan_edge,regularization_loss = self.get_trojan(self.prompt.promptEdges,self.prompt.shared_prompt_pool,self.prompt.shared_edge_weights,idx_outter,features,self.edge_index,target_classes = idx_outter_target_classes,mode='R')
                    elif item == 'F':
                        trojan_feat, trojan_weights,trojan_edge,regularization_loss = self.get_trojan(self.prompt.promptEdges,self.prompt.shared_prompt_pool,self.prompt.shared_edge_weights,idx_outter,features,self.edge_index,target_classes = idx_outter_target_classes)
                
                    trojan_edge = trojan_edge.reshape(edge_index.shape[0],-1)   
                    
                    if args.position:
                        regularization_loss = regularization_loss
                    
                 
                    poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]) # repeat trojan weights beacuse of undirected edge
                    poison_x = torch.cat([features,trojan_feat])
                    poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)
                    
                    if args.prompt_type == 'GPF':
                        poison_x = self.prompt_origin.add(poison_x)
                    elif args.prompt_type == 'All-in-one':
                        pg_data = Data(x=poison_x, edge_index=poison_edge_index)
                        pg_poison_x, pg_poison_edge_index= self.prompt_origin(pg_data,idx_outter)
                        poison_edge_weights = torch.cat([torch.ones([(pg_poison_edge_index.shape[1]-poison_edge_index.shape[1])//2],dtype=torch.float, device=device),poison_edge_weights]).to(device)
                    if args.prompt_type == 'All-in-one':
                        embedding = self.surrogate_model(pg_poison_x, pg_poison_edge_index,poison_edge_weights)
                    else:
                        embedding = self.surrogate_model(poison_x, poison_edge_index,poison_edge_weights)
                    if args.prompt_type == 'GPPT':
                        output = self.prompt_origin(embedding, poison_edge_index)
                    elif args.prompt_type == 'GPF' or args.prompt_type == 'All-in-one':
                        output = self.answering(embedding)
                    elif args.prompt_type == 'Gprompt':
                        embedding = self.prompt_origin(embedding)
                        # embedding = global_mean_pool(embedding)
                        output = self.answering(embedding)
                        
                    labels_outter = labels.clone()
                    labels_outter[idx_outter] = idx_outter_target_classes
                    
                    loss_target = self.loss(output[torch.cat([idx_train,idx_outter])], labels_outter[torch.cat([idx_train,idx_outter])]) 
                    acc_outter =(output[idx_outter].argmax(dim=1)==idx_outter_target_classes).float().mean()
                    loss_homo = 0.0
                    if(self.args.homo_loss_weight > 0):
                        loss_homo  = self.homoss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
                                                    trojan_weights,\
                                                    poison_x,\
                                                    self.args.homo_boost_thrd)
                    
                    loss_outter = loss_target + self.args.homo_loss_weight *loss_homo
                    
                    if args.finetune == True:
                        loss_outter += self.args.norm_weight *regularization_loss 
                    
                    loss_outter.backward(retain_graph=True) 
                
                
                optimizer_prompt_pool.step()
                optimizer_homo.step()
                optimizer_selector.step() 
                self.prompt.update()
           
                if (i+1) % 10 == 0:
                    
                    logging.info('Epoch:{},acc_train:{:.2f},,acc_attach:{:.2f},cos_loss:{:.4f}'.format(i+1,acc_train,acc_train_attach,loss_homo))          
    
                    logging.info('loss_inner:{:.4f},loss_outter:{:.4f},acc_outter:{:.4f}'.format(loss_inner,loss_target,acc_outter))
            
              
                if self.args.val_feq:
                    if (i+1) % self.args.val_feq == 0:
                        self.prompt.eval()
                        self.homo.eval()
                        self.surrogate_model.eval()
                        self.prompt_origin.eval()
                        from utils import prune_unrelated_edge
                        
                        poison_x_list = []
                        poison_edge_list = []
                        poison_weights_list = []
                        poison_labels_list = []
                        
                    
                        if self.args.fit_attach_num<=len(idx_attach):
                            idx_attach_val = idx_attach[:self.args.fit_attach_num]
                        else:
                            idx_attach_val = idx_attach
                   
                        if args.attack_single:
                            eval_feat, eval_edge,eval_weights,_,_,_= self.get_attach_poisoned(args.target_label,labels,idx_attach_val)
                        else:
                            eval_feat, eval_edge,eval_weights,_,_,_= self.get_attach_poisoned(None,labels,idx_attach_val)
                    
                        # eval_edge,eval_weights = prune_unrelated_edge(args,eval_edge,eval_weights,eval_feat,device,large_graph=False)
                
                        eval_edge = eval_edge.reshape(edge_index.shape[0],-1)
                        
                        
                    
                        if self.args.fit_attach_num<=len(idx_attach):
                            new_idx_attach = idx_attach[:self.args.fit_attach_num]
                        else:
                            new_idx_attach = idx_attach
                      
                        
                        if args.attack_single:
                            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(args.target_label,labels,new_idx_attach)
                        else:
                            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(None,labels,new_idx_attach)
                    
                        
                        poison_x_list.append(poison_x)
                        poison_edge_list.append(poison_edge_index)
                        poison_weights_list.append(poison_edge_weights)
                        poison_labels_list.append(poison_labels)
                        
                        
                        self.val_model = copy.deepcopy(self.surrogate_model)
                        if args.prompt_type == 'GPPT':
                            self.val_pg = GPPTPrompt(args.hidden, labels.max().item() + 1, labels.max().item() + 1, device = device)
                            node_embedding = self.val_model(features, edge_index)
                            self.val_pg.weigth_init(node_embedding,edge_index, labels, idx_train)
                            self.val_pg.new_fit(args,self.val_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_val,train_iters=1000, verbose=False)
    
                        elif args.prompt_type == 'GPF':
                            self.val_pg = GPF(features.shape[1]).to(device)
                            self.val_answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                                            torch.nn.Softmax(dim=1)).to(device)
                            self.val_pg.new_fit(args,self.val_answering,self.val_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_val,train_iters=1000, verbose=False) 
                        elif args.prompt_type == 'Gprompt':
                            self.val_pg = Gprompt(args.hidden).to(device)
                            self.val_answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                                            torch.nn.Softmax(dim=1)).to(device)
                            self.val_pg.new_fit(args,self.val_answering,self.val_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_val,train_iters=1000, verbose=False) 
                        elif args.prompt_type ==  'All-in-one':
                            self.val_pg = HeavyPrompt(token_dim=features.shape[1], token_num=3, cross_prune=0.1, inner_prune=0.3).to(device)
                            self.val_answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                            torch.nn.Softmax(dim=1)).to(device)
                            self.val_pg.new_fit(args,self.val_answering,self.val_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_val,train_iters=1000, verbose=False) 
            
                            
                        if args.position:
                            pass
                        else:
                            eval_feat = self.getFineTune(eval_feat)
                        
                        eval_edge_weights = torch.cat([edge_weight,eval_weights,eval_weights]) # repeat trojan weights beacuse of undirected edge
                        eval_x = torch.cat([features,eval_feat]).to(device)
                        # if args.prompt_type == 'GPF':
                        #     eval_x = self.prompt_origin.add(eval_x)
                        
                        eval_edge_index = torch.cat([edge_index,eval_edge],dim=1).to(device)
                        overall_induct_edge_index = eval_edge_index.clone()
                        asr = 0
                      
                        for _, idx in enumerate(idx_val):
                            idx=int(idx)
                            sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                            
                            ori_node_idx = sub_induct_nodeset[sub_mapping]
                            relabeled_node_idx = sub_mapping
                            sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                            asr_val = 0
                            
                            for index in range(args.attack_num):
                                # index = index + 1
                                # index = self.target_labels[q]
                                # if index == 2:
                                #     index = 3
                                # elif index == 1:
                                #     index = 2
                                if self.args.attack_single:
                                    index = self.args.target_label
                                with torch.no_grad():
                                    induct_x, induct_edge_index,induct_edge_weights,_,_,target_class = self.get_attach_poision(relabeled_node_idx,eval_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device,index)
                                    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                                    if args.prompt_type == 'GPF':
                                        induct_x = self.val_pg.add(induct_x)
                        
                                    if args.mode:
                                        if args.prompt_type == 'All-in-one':
                                            pg_data = Data(x=induct_x,edge_index=induct_edge_index)
                                            pg_induct_x, pg_induct_edge_index= self.val_pg(pg_data,relabeled_node_idx)
                                        if args.prompt_type == 'All-in-one':
                                            output = self.val_model(pg_induct_x,pg_induct_edge_index,None)
                                        else:
                                            output = self.val_model(induct_x,induct_edge_index,None)
                                    else:
                                        if args.prompt_type == 'All-in-one':
                                            pg_data = Data(x=induct_x,edge_index=induct_edge_index)
                                            
                                            pg_induct_x, pg_induct_edge_index= self.val_pg(pg_data,relabeled_node_idx)
                                            induct_edge_weights = torch.cat([torch.ones([(pg_induct_edge_index.shape[1]-induct_edge_index.shape[1])//2],dtype=torch.float, device=device),induct_edge_weights]).to(device)
                                            # induct_edge_weights = torch.cat(torch.ones((pg_induct_edge_index.shape[1]-induct_edge_index.shape[1])//2),induct_edge_weights).to(device)
                                        if args.prompt_type == 'All-in-one':
                                            output = self.val_model(pg_induct_x,pg_induct_edge_index,induct_edge_weights)
                                        else:
                                            output = self.val_model(induct_x,induct_edge_index,induct_edge_weights)
                                        # induct_edge_index,induct_edge_weights= prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
                                        
                                        # output = self.val_model(induct_x,induct_edge_index,induct_edge_weights)
                                    
                                    if args.prompt_type == 'GPPT':
                                        output = self.val_pg(output, induct_edge_index)
                                    elif args.prompt_type == 'GPF' or args.prompt_type == 'All-in-one':
                                        output = self.val_answering(output)
                                    elif args.prompt_type == 'Gprompt':
                                        embedding = self.val_pg(output)
                                        # output = global_mean_pool(embedding)
                                        output = self.val_answering(embedding)
                                        
                                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==target_class).float().mean()
                                    asr_val+=train_attach_rate
                                
                            
                            asr_val = asr_val/args.attack_num
                            asr += asr_val
                        asr = asr/(idx_val.shape[0])
                        acc_val = asr
                        logging.info('Epoch:{},val_acc:{:.4f}'.format(i+1,acc_val))
                    # if (i+1) % self.args.val_feq == 0:
                        if  acc_val.item()> loss_best :
                            loss_best = acc_val.item()
                            label_full = self.label_full
                            label_origin = self.label_origin
                            orginLabels = self.orginLabels
                            val_idx_attach = self.idx_attach
                            weights = deepcopy(self.prompt.state_dict())
                            homo_weights = deepcopy(self.homo.state_dict())
                            total_weights = {'weights': weights, 'homo_weights': homo_weights,'label_full':label_full,'label_origin':label_origin,'orginLabels':orginLabels,'val_idx_attach':val_idx_attach}  
                            torch.save(total_weights, './parameters/{}_{}.pth'.format(self.args.dataset,self.args.index))
                        
                            early_stop_counter = 0  
                
            print(f"Best val acc: {loss_best:.4f}",flush=True)
        
        if args.train == False:
            print('==test begin==')
            file_path = './parameters/{}_{}.pth'.format(args.dataset,args.index)
            total_weights = torch.load(file_path)  
            weights = total_weights['weights']
            homo_weights = total_weights['homo_weights']
            label_full = total_weights['label_full']
            label_origin = total_weights['label_origin']
            orginLabels = total_weights['orginLabels']
            val_idx_attach = total_weights['val_idx_attach']
        
        idx_attach = val_idx_attach
        self.idx_attach = idx_attach
        self.label_full =  label_full 
        self.label_origin = label_origin 
        self.orginLabels = orginLabels 
        self.prompt.load_state_dict(weights)
        self.homo.load_state_dict(homo_weights)
        
        self.prompt.eval()
        self.homo.eval()  
         
        if args.train:
            # self.features = ori_features
            if args.down_method == 'GSL':
                print('GSL',flush=True)
                self.GSL_test(idx_train,idx_attach,idx_val,idx_atk,self.data,self.data.y.clone(),idx_clean_test,mask_edge_index)
            elif args.down_method == 'GCL':
                print('GCL',flush=True)
                self.GCL_test(idx_train,idx_attach,idx_val,idx_atk,self.data,self.data.y.clone(),idx_clean_test,mask_edge_index)
            else:
                print('GPL',flush=True)
                self.test(idx_train,idx_attach,idx_val,idx_atk,self.data,self.data.y.clone(),idx_clean_test,mask_edge_index)
           
        # if args.train == False:
        #     self.new_test(idx_train,idx_val,idx_atk,self.data,self.data.y.clone(),idx_clean_test,mask_edge_index)
        
    def get_attach_poisoned(self,target,labels,new_idx_attach):
        
        new_labels = labels.clone()
        with torch.no_grad():
            if target is not None:
                # poison_x, poison_edge_index, poison_edge_weights,trojan_feat,trojan_edge,_ = self.get_attach_poision(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device,target)
                # new_labels[self.idx_attach] = target
                poison_x, poison_edge_index, poison_edge_weights,trojan_feat,trojan_edge,_ = self.get_attach_poision(new_idx_attach,self.features,self.edge_index,self.edge_weights,self.device,target)
                if len(new_idx_attach) != 0:
                    new_labels[new_idx_attach] = target
                
            else:
                # poison_x, poison_edge_index, poison_edge_weights,trojan_feat,trojan_edge,target_classes = self.get_random_attach_poision(self.idx_attach, labels,self.features,self.edge_index,self.edge_weights,self.device)
                # new_labels[self.idx_attach] = target_classes
                poison_x, poison_edge_index, poison_edge_weights,trojan_feat,trojan_edge,target_classes = self.get_random_attach_poision(new_idx_attach, labels,self.features,self.edge_index,self.edge_weights,self.device)
                if len(new_idx_attach) != 0:
                    new_labels[new_idx_attach] = target_classes
        
        poison_labels = new_labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        # print('fixed pool:',self.prompt.shared_prompt_pool,self.prompt.shared_prompt_pool.shape)
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels,trojan_feat,trojan_edge
    
    def GSL_test(self,idx_train,idx_attach,idx_eval,idx_atk,data,labels,idx_clean_test,mask_edge_index):
        loss_labels = calculate_weights(idx_train,labels)
        loss_labels = torch.tensor(loss_labels).to(self.device)

        args = self.args
        device =self.device
        data.x =  self.features
      
        if self.args.fit_attach_num<=len(idx_attach):
            idx_attach = idx_attach[:self.args.fit_attach_num]
        else:
            idx_attach = idx_attach
        
        if self.args.attack_single:
            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(self.args.target_label,labels,idx_attach)
        else:
            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(None,labels,idx_attach)
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= self.get_attach_poisoned(0,labels,idx_attach)
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= self.get_attach_poisoned(None,labels,idx_attach)
        # weights = {
        #     'poison_x':poison_x,
        #     'poison_edge_index':poison_edge_index,
        #     'poison_edge_weights':poison_edge_weights,
        #     'poison_labels':poison_labels,
        #     'idx_attach':idx_attach,
        # }
        # file_path = './pg_test/{}/origin-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        # if os.path.exists(file_path):  
        #     pass
        # else: 
        #     torch.save(weights,file_path)
        
        
        origin_poison_x = poison_x
        origin_poison_edge_index = poison_edge_index
        origin_poison_edge_weights = poison_edge_weights
        origin_poison_labels = poison_labels
        origin_idx_attach = idx_attach
      
        poison_x_list = []
        poison_edge_list = []
        poison_weights_list = []
        poison_labels_list = []
        poison_x_list.append(deepcopy(poison_x))
        poison_edge_list.append(deepcopy(poison_edge_index))
        poison_weights_list.append(deepcopy(poison_edge_weights))
        poison_labels_list.append(deepcopy(poison_labels))
        # fit_weight = {
        #     'poison_x_list':poison_x_list,
        #     'poison_edge_list':poison_edge_list,
        #     'poison_weights_list':poison_weights_list,
        #     'poison_labels_list':poison_labels_list,
        # }
     
        # file_path = './pg_test/{}/fit-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        # if os.path.exists(file_path):  
        #     pass
        # else:  
        #     torch.save(fit_weight,file_path)
     
        rs = np.random.RandomState(args.seed)
        seeds = rs.randint(1000,size=10)
        seeds = list(seeds)
        seeds.insert(0, args.seed) 
       
        
        
        models = ['GCN','GAT', 'GraphSage'] #'GCN','GAT', 'GraphSage'
        defense_modes =  ['none','reconstruct',"prune"]
        for seed in seeds:
        
            args.seed = seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

            for defense_mode in defense_modes:
                args.defense_mode = defense_mode
                
                idx_attach = origin_idx_attach
                
                bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(self.device)
                if(args.defense_mode == 'prune'):
                    poison_edge_index,poison_edge_weights = prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,self.device,large_graph=False)
                elif(args.defense_mode == 'isolate'):
                    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,device,large_graph=False)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
                    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
                    idx_attach = torch.LongTensor(list(set(idx_attach) - set(rel_nodes))).to(device)
                elif(args.defense_mode == 'reconstruct'):
                    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,data.x,data.edge_index,self.device, idx_attach, large_graph=True)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
                if self.args.fit_attach_num<=len(idx_attach):
                    new_idx_attach = idx_attach[:self.args.fit_attach_num]
                else:
                    new_idx_attach = idx_attach
                    
                
                for test_model in models:
                    args.test_model = test_model
                    asr = 0
                    
                    test_model = model_construct(args,args.test_model,data,device).to(device)
                    
                    
                    test_model.new_fit(poison_x_list, poison_edge_list, poison_weights_list, poison_labels_list, [idx_train,new_idx_attach], idx_eval, None,train_iters=1000, verbose=False)
                    
                        
                    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
                    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
                    # poison_edge_index,poison_x,poison_edge_weights= model.test(idx_atk,idx_attach,features,edge_index)
                    clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

                    print("accuracy on clean test nodes: {:.4f}".format(clean_acc), flush=True)
                    

                    from torch_geometric.utils  import k_hop_subgraph
                    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
                    
                    
                    for index in range(args.attack_num):
                        induct_x_list = []  
                        induct_edge_index_list = []  
                        induct_edge_weights_list = []  
                        trojan_edge_index_list = []  
                        target_class_list = []  
                        relabeled_node_idx_list = [] 
                        if args.attack_single:
                            index = args.target_label
                        
                        for i, idx in enumerate(idx_atk):
                            
                            idx=int(idx)
                            sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                            
                            ori_node_idx = sub_induct_nodeset[sub_mapping]
                            relabeled_node_idx = sub_mapping
                            sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                            
                        
                            with torch.no_grad():
                                induct_x, induct_edge_index,induct_edge_weights,_,trojan_edge_index,target_class = self.get_attach_poision(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device,index)
                                induct_x_list.append(induct_x)
                                induct_edge_index_list.append(induct_edge_index)
                                induct_edge_weights_list.append(induct_edge_weights)
                                trojan_edge_index_list.append(trojan_edge_index) 
                                target_class_list.append(target_class)  
                                relabeled_node_idx_list.append(relabeled_node_idx)
                                
                                induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                            
                                if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                                    origin_edge_index = induct_edge_index
                                
                                    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
                                    normal_edges_count,special_edges_count = count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index)
                                    # count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index, origin_edge_index)
                                    # normal_edges_count,special_edges_count = count_edges(old_edge_index,induct_edge_index,total_nodes)
                                    # total_special_edges_count += special_edges_count
                                    # total_normal_edges_count += normal_edges_count
                                    
                                output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                                    
                                # train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                                train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==target_class).float().mean()
                                
                                asr +=train_attach_rate
                                induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                                output = output.cpu()
                        
                        
                    asr = asr/(idx_atk.shape[0])/args.attack_num
                    print("Defense: {},Test_model: {},Overall ASR: {:.4f}".format(args.defense_mode,args.test_model,asr), flush=True)
                    
        print('==test over==', flush=True)
        
    def GCL_test(self,idx_train,idx_attach,idx_eval,idx_atk,data,labels,idx_clean_test,mask_edge_index):
        
        loss_labels = calculate_weights(idx_train,labels)
        loss_labels = torch.tensor(loss_labels).to(self.device)
        

        args = self.args
        device =self.device
        
       
        if self.args.fit_attach_num<=len(idx_attach):
            idx_attach = idx_attach[:self.args.fit_attach_num]
        else:
            idx_attach = idx_attach
        
        if self.args.attack_single:
            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(self.args.target_label,labels,idx_attach)
        else:
            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(None,labels,idx_attach)
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= self.get_attach_poisoned(0,labels,idx_attach)
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= self.get_attach_poisoned(None,labels,idx_attach)
        # weights = {
        #     'poison_x':poison_x,
        #     'poison_edge_index':poison_edge_index,
        #     'poison_edge_weights':poison_edge_weights,
        #     'poison_labels':poison_labels,
        #     'idx_attach':idx_attach,
        # }
        # file_path = './test/{}/origin-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        # if os.path.exists(file_path):  
        #     pass
        # else: 
        #     torch.save(weights,file_path)
        
        
        origin_poison_x = poison_x
        origin_poison_edge_index = poison_edge_index
        origin_poison_edge_weights = poison_edge_weights
        origin_poison_labels = poison_labels
        origin_idx_attach = idx_attach
        
        poison_x_list = []
        poison_edge_list = []
        poison_weights_list = []
        poison_labels_list = []
        poison_x_list.append(deepcopy(poison_x))
        poison_edge_list.append(deepcopy(poison_edge_index))
        poison_weights_list.append(deepcopy(poison_edge_weights))
        poison_labels_list.append(deepcopy(poison_labels))
        # fit_weight = {
        #     'poison_x_list':poison_x_list,
        #     'poison_edge_list':poison_edge_list,
        #     'poison_weights_list':poison_weights_list,
        #     'poison_labels_list':poison_labels_list,
        # }
     
        # file_path = './test/{}/fit-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        # if os.path.exists(file_path):  
        #     pass
        # else:  
        #     torch.save(fit_weight,file_path)
        
        rs = np.random.RandomState(args.seed)
        seeds = rs.randint(1000,size=3) #10
        seeds = list(seeds)
        seeds.insert(0, args.seed) 
        
        
        encoder = Encoder(self.features.shape[1], self.args.gcl_hidden ,k=2).to(device)
        clean_model = GRACE(encoder = encoder,
                        nfeat=self.features.shape[1],
                        nhid=self.args.gcl_hidden,
                        nclass=labels.max().item() + 1,
                        layer = 2,
                        dropout=0.5, device=self.device,args = self.args,method = 'GRACE').to(self.device)
        # GRACE
        # clean_model.new_fit([self.features], [self.edge_index], [None], [labels], [idx_train,None], idx_eval, None,train_iters=100, verbose=None)
        
        # CCA-SSG
        clean_model.new_fit([self.features], [self.edge_index], [None], [labels], [idx_train,None], idx_eval, None,train_iters=1000, verbose=None)
        
        if args.dataset != 'Cora':
            self.nodes_idx =  label_num(idx_train,labels)
            idx_clean = random_select(self.nodes_idx,top_k = self.args.top_k).to(self.device)
            idx_train = idx_clean
        
        if args.dataset == 'Cora':
            classifer = train_classifer_cora(clean_model,(idx_train,None),idx_eval,self.features,self.edge_index,labels,device,args,train_iters=1000, verbose=False)
        else:
            classifer = train_classifer(clean_model,(idx_train,None),idx_eval,self.features,self.edge_index,labels,device,args,train_iters=1000, verbose=False)
        clean_acc = clean_model.test(classifer,self.features,self.edge_index,None,data.y,idx_clean_test)
        print("clean encoder / clean classifer--acc: {:.4f}".format(clean_acc), flush=True)
        
        clean_model.new_fit(poison_x_list, poison_edge_list, poison_weights_list, poison_labels_list, [idx_train,idx_attach], idx_eval, None,train_iters=1000, verbose=False)

        if args.dataset == 'Cora':
            classifer = train_classifer_cora(clean_model,(idx_train,None),idx_eval,self.features,self.edge_index,labels,device,args,train_iters=1000, verbose=False)
        else:
            classifer = train_classifer(clean_model,(idx_train,None),idx_eval,self.features,self.edge_index,labels,device,args,train_iters=1000, verbose=False)
        # classifer = train_classifer(clean_model,(idx_train,None),idx_eval,self.features,self.edge_index,labels,device,args,train_iters=1000, verbose=False)
        clean_acc = clean_model.test(classifer,self.features,self.edge_index,None,data.y,idx_clean_test)
        print("posioned encoder / clean classifer--acc: {:.4f}".format(clean_acc), flush=True)
        
        models = ['GRACE'] #'GCN','GAT', 'GraphSage'
        defense_modes = ['none', 'prune','reconstruct'] #
        # defense_modes = ['reconstruct']
        # reconstruct
        for seed in seeds:
            
            args.seed = seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

            for defense_mode in defense_modes:
                args.defense_mode = defense_mode
                
                idx_attach = origin_idx_attach
                
                bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(self.device)
                if(args.defense_mode == 'prune'):
                    poison_edge_index,poison_edge_weights = prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,self.device,large_graph=False)
                elif(args.defense_mode == 'isolate'):
                    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,device,large_graph=False)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
                    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
                    idx_attach = torch.LongTensor(list(set(idx_attach) - set(rel_nodes))).to(device)
                elif(args.defense_mode == 'reconstruct'):
                    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,data.x,data.edge_index,self.device, idx_attach, large_graph=True)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
                if self.args.fit_attach_num<=len(idx_attach):
                    new_idx_attach = idx_attach[:self.args.fit_attach_num]
                else:
                    new_idx_attach = idx_attach
                    
                
                for test_model in models:
                    args.test_model = test_model
                    asr = 0
                    
                    
                   
                    # test_model = model_construct(args,args.test_model,data,device).to(device)
                    
                   
                    test_model = copy.deepcopy(clean_model)
                    # TODO
                    # test_classifer = train_classifer(test_model,(idx_train,None),idx_eval,self.features,self.edge_index,labels,device,args,train_iters=1000, verbose=False)
                    # test_model.new_fit(poison_x_list, poison_edge_list, poison_weights_list, poison_labels_list, [idx_train,new_idx_attach], idx_eval, None,train_iters=300, verbose=False)
                    if args.dataset == 'Cora':
                        test_classifer = train_classifer_cora(test_model,(idx_train,new_idx_attach),idx_eval,poison_x_list[0],poison_edge_list[0],poison_labels_list[0],device,args,train_iters=1000, verbose=False)
                    else:
                        test_classifer = train_classifer(test_model,(idx_train,new_idx_attach),idx_eval,poison_x_list[0],poison_edge_list[0],poison_labels_list[0],device,args,train_iters=1000, verbose=False)
                    # test_classifer = train_classifer(test_model,(idx_train,new_idx_attach),idx_eval,poison_x_list[0],poison_edge_list[0],poison_labels_list[0],device,args,train_iters=1000, verbose=False,with_val=True)
                    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
                    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
                    # poison_edge_index,poison_x,poison_edge_weights= model.test(idx_atk,idx_attach,features,edge_index)
                    clean_acc = test_model.test(test_classifer,poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

                    print("accuracy on clean test nodes: {:.4f}".format(clean_acc), flush=True)
                    

                    from torch_geometric.utils  import k_hop_subgraph
                    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
                   
                    # TODO
                    # for index in range(1):
                    for index in range(self.args.attack_num):
                        label_acc = 0
                    # for index in range(labels.max().item()+1):
                        if self.args.attack_single:
                            index = self.args.target_label
                        induct_x_list = []  
                        induct_edge_index_list = []  
                        induct_edge_weights_list = []  
                        trojan_edge_index_list = []  
                        target_class_list = []  
                        relabeled_node_idx_list = [] 
                        
                        
                        for i, idx in enumerate(idx_atk):
                            
                            idx=int(idx)
                            sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                            
                            ori_node_idx = sub_induct_nodeset[sub_mapping]
                            relabeled_node_idx = sub_mapping
                            sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                            
                        
                            with torch.no_grad():
                                induct_x, induct_edge_index,induct_edge_weights,_,trojan_edge_index,target_class = self.get_attach_poision(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device,index)
                                induct_x_list.append(induct_x)
                                induct_edge_index_list.append(induct_edge_index)
                                induct_edge_weights_list.append(induct_edge_weights)
                                trojan_edge_index_list.append(trojan_edge_index) 
                                target_class_list.append(target_class)  
                                relabeled_node_idx_list.append(relabeled_node_idx)
                                
                                induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                            
                                if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                                    origin_edge_index = induct_edge_index
                                
                                    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
                                    normal_edges_count,special_edges_count = count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index)
                                    # count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index, origin_edge_index)
                                    # normal_edges_count,special_edges_count = count_edges(old_edge_index,induct_edge_index,total_nodes)
                                    # total_special_edges_count += special_edges_count
                                    # total_normal_edges_count += normal_edges_count
                                    
                                output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                                output = test_classifer(output)
                                # train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                                train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==target_class).float().mean()
                                
                                asr +=train_attach_rate
                                label_acc += train_attach_rate
                                induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                                output = output.cpu()
                        
                        # file_path = './test/{}/label{}-{}-{}-{}-{}.pth'.format(args.dataset,index,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
                        # if os.path.exists(file_path):
                        #     pass
                        # else:
                        #     sub_data = {
                        #         'induct_x_list':induct_x_list,
                        #         'induct_edge_index_list':induct_edge_index_list,  
                        #         'induct_edge_weights_list':induct_edge_weights_list, 
                        #         'trojan_edge_index_list':trojan_edge_index_list,
                        #         'target_class_list':target_class_list,
                        #         'relabeled_node_idx_list':relabeled_node_idx_list,
                        #     }
                        #     torch.save(sub_data,file_path)
                        
                        print('label{}:{:.4f}'.format(index,label_acc/(idx_atk.shape[0])),flush=True)
                    # TODO
                    asr = asr/(idx_atk.shape[0])/self.args.attack_num
                    # asr = asr/(idx_atk.shape[0])
                    # asr = asr/(idx_atk.shape[0])/(labels.max().item()+1)
                    print("Defense: {},Test_model: {},Overall ASR: {:.4f}".format(args.defense_mode,args.test_model,asr), flush=True)
                    
        print('==test over==', flush=True)
    
    def test(self,idx_train,idx_attach,idx_eval,idx_atk,data,labels,idx_clean_test,mask_edge_index):
        loss_labels = calculate_weights(idx_train,labels)
        loss_labels = torch.tensor(loss_labels).to(self.device)

        args = self.args
        device =self.device
        
       
        if self.args.fit_attach_num<=len(idx_attach):
            idx_attach = idx_attach[:self.args.fit_attach_num]
        else:
            idx_attach = idx_attach
        
        if self.args.attack_single:
            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(self.args.target_label,labels,idx_attach)
        else:
            poison_x, poison_edge_index, poison_edge_weights, poison_labels,_,_=self.get_attach_poisoned(None,labels,idx_attach)
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= self.get_attach_poisoned(0,labels,idx_attach)
        # poison_x, poison_edge_index,poison_edge_weights,poison_labels,_,_= self.get_attach_poisoned(None,labels,idx_attach)
        # weights = {
        #     'poison_x':poison_x,
        #     'poison_edge_index':poison_edge_index,
        #     'poison_edge_weights':poison_edge_weights,
        #     'poison_labels':poison_labels,
        #     'idx_attach':idx_attach,
        # }
        # file_path = './pg_test/{}/origin-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        # if os.path.exists(file_path):  
        #     pass
        # else: 
        #     torch.save(weights,file_path)
        
        
        origin_poison_x = poison_x
        origin_poison_edge_index = poison_edge_index
        origin_poison_edge_weights = poison_edge_weights
        origin_poison_labels = poison_labels
        origin_idx_attach = idx_attach
        
        poison_x_list = []
        poison_edge_list = []
        poison_weights_list = []
        poison_labels_list = []
        poison_x_list.append(deepcopy(poison_x))
        poison_edge_list.append(deepcopy(poison_edge_index))
        poison_weights_list.append(deepcopy(poison_edge_weights))
        poison_labels_list.append(deepcopy(poison_labels))
        # fit_weight = {
        #     'poison_x_list':poison_x_list,
        #     'poison_edge_list':poison_edge_list,
        #     'poison_weights_list':poison_weights_list,
        #     'poison_labels_list':poison_labels_list,
        # }
     
        # file_path = './pg_test/{}/fit-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        # if os.path.exists(file_path):  
        #     pass
        # else:  
        #     torch.save(fit_weight,file_path)
        
        rs = np.random.RandomState(args.seed)
        seeds = rs.randint(1000,size=10)
        seeds = list(seeds)
        seeds.insert(0, args.seed) 
        
        
        
        models = ['GRACE'] #'GCN','GAT', 'GraphSage'
        defense_modes = ['none']
        # defense_modes = ['reconstruct']
        # reconstruct
        for seed in seeds:
            
            args.seed = seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

            for defense_mode in defense_modes:
                args.defense_mode = defense_mode
                
                idx_attach = origin_idx_attach
                
                bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(self.device)
                if(args.defense_mode == 'prune'):
                    poison_edge_index,poison_edge_weights = prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,self.device,large_graph=False)
                elif(args.defense_mode == 'isolate'):
                    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,device,large_graph=False)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
                    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
                    idx_attach = torch.LongTensor(list(set(idx_attach) - set(rel_nodes))).to(device)
                elif(args.defense_mode == 'reconstruct'):
                    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,data.x,data.edge_index,self.device, idx_attach, large_graph=True)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
                if self.args.fit_attach_num<=len(idx_attach):
                    new_idx_attach = idx_attach[:self.args.fit_attach_num]
                else:
                    new_idx_attach = idx_attach
                    
                
                for test_model in models:
                    args.test_model = test_model
                    asr = 0
                    
                    
                    
                    # test_model = model_construct(args,args.test_model,data,device).to(device)
                   
                    test_model = copy.deepcopy(self.surrogate_model)
                    if args.prompt_type == 'GPPT':
                        test_pg = GPPTPrompt(args.hidden, labels.max().item() + 1, labels.max().item() + 1, device = device)
                        node_embedding = test_model(self.features, self.edge_index)
                        test_pg.weigth_init(node_embedding,self.edge_index, labels, idx_train)
                        test_pg.new_fit(args,test_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_eval,train_iters=1000, verbose=False)
                    elif args.prompt_type == 'GPF':
                        test_pg = GPF(self.features.shape[1]).to(device)
                        test_answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                                        torch.nn.Softmax(dim=1)).to(device)
                        test_pg.new_fit(args,test_answering,test_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_eval,train_iters=1000, verbose=False) #TODO
                    
                    elif args.prompt_type == 'Gprompt':
                        test_pg = Gprompt(args.hidden).to(device)
                        test_answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                                        torch.nn.Softmax(dim=1)).to(device)
                        test_pg.new_fit(args,test_answering,test_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_eval,train_iters=1000, verbose=False) #TODO
                    elif args.prompt_type == 'All-in-one':
                        test_pg = HeavyPrompt(token_dim=self.features.shape[1], token_num=3, cross_prune=0.1, inner_prune=0.3).to(device)
                        test_answering =  torch.nn.Sequential(torch.nn.Linear(args.hidden,labels.max().item() + 1),
                                                        torch.nn.Softmax(dim=1)).to(device) 
                        test_pg.new_fit(args,test_answering,test_model,poison_x_list[0], poison_edge_list[0], poison_weights_list[0], poison_labels_list[0], [idx_train,new_idx_attach], idx_eval,train_iters=1000, verbose=False) #TODO
                    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
                    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
                    
                    
                    if args.prompt_type == 'GPPT':
                        clean_acc = test_pg.test(test_model,poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
                    elif args.prompt_type == 'GPF':
                        clean_acc = test_pg.test(test_answering,test_model,poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
                    elif args.prompt_type == 'All-in-one':
                        clean_acc = test_pg.test(test_answering,test_model,poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
                    elif args.prompt_type == 'Gprompt':
                        clean_acc = test_pg.test(test_answering,test_model,poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)# TODO
                    print("accuracy on clean test nodes: {:.4f}".format(clean_acc), flush=True)
                    

                    from torch_geometric.utils  import k_hop_subgraph
                    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
                   
                    
                    # for index in range(1):
                    for index in range(self.args.attack_num):
                        label_acc = 0
                        # index = self.target_labels[index]
                        # index = index + 1
                        # if index == 2:
                        #     index = 3
                        # elif index == 1:
                        #     index = 2
                    # # for index in range(labels.max().item()+1):
                        if self.args.attack_single:
                            index = self.args.target_label
                        induct_x_list = []  
                        induct_edge_index_list = []  
                        induct_edge_weights_list = []  
                        trojan_edge_index_list = []  
                        target_class_list = []  
                        relabeled_node_idx_list = [] 
                        
                        record_features_list = []
                        for i, idx in enumerate(idx_atk):
                            
                            idx=int(idx)
                            sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                            
                            ori_node_idx = sub_induct_nodeset[sub_mapping]
                            relabeled_node_idx = sub_mapping
                            sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                            
                        
                            with torch.no_grad():
                                induct_x, induct_edge_index,induct_edge_weights,_,trojan_edge_index,target_class = self.get_attach_poision(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device,index)
                                induct_x_list.append(induct_x)
                                induct_edge_index_list.append(induct_edge_index)
                                induct_edge_weights_list.append(induct_edge_weights)
                                trojan_edge_index_list.append(trojan_edge_index) 
                                target_class_list.append(target_class)  
                                relabeled_node_idx_list.append(relabeled_node_idx)
                                
                                induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                                if args.prompt_type == 'GPF':
                                    induct_x = test_pg.add(induct_x)
                                elif args.prompt_type == 'All-in-one':
                                    pg_data = Data(x=induct_x,edge_index=induct_edge_index)
                                    pg_induct_x, pg_induct_edge_index= test_pg(pg_data,relabeled_node_idx)
                                    induct_edge_weights = torch.cat([torch.ones([(pg_induct_edge_index.shape[1]-induct_edge_index.shape[1])//2],dtype=torch.float, device=device),induct_edge_weights]).to(device)
                                    # induct_edge_weights = torch.cat(torch.ones((pg_induct_edge_index.shape[1]-induct_edge_index.shape[1])//2),induct_edge_weights).to(device)
                                if args.prompt_type == 'All-in-one':
                                    output = test_model(pg_induct_x,pg_induct_edge_index,induct_edge_weights)
                                else:
                                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                                    
                                if args.prompt_type == 'GPPT':
                                    output = test_pg(output,induct_edge_index)
                                elif args.prompt_type == 'GPF' or args.prompt_type == 'All-in-one':
                                    output = test_answering(output)
                                elif args.prompt_type == 'Gprompt':
                                    embedding = test_pg(output)
                                    # output = global_mean_pool(embedding)
                                    output = test_answering(embedding)
                                train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==target_class).float().mean()
                                
                                asr +=train_attach_rate
                                label_acc += train_attach_rate
                                induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                                output = output.cpu()
                                record_features_list.append(output)
                        # file_path = './pg_test/{}/label{}-{}-{}-{}-{}.pth'.format(args.dataset,index,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
                        # if os.path.exists(file_path):
                        #     pass
                        # else:
                        #     sub_data = {
                        #         'induct_x_list':induct_x_list,
                        #         'induct_edge_index_list':induct_edge_index_list,  
                        #         'induct_edge_weights_list':induct_edge_weights_list, 
                        #         'trojan_edge_index_list':trojan_edge_index_list,
                        #         'target_class_list':target_class_list,
                        #         'relabeled_node_idx_list':relabeled_node_idx_list,
                        #         'record_features_list':record_features_list,
                        #     }
                        #     torch.save(sub_data,file_path)
                        
                        print('label{}:{:.4f}'.format(index,label_acc/(idx_atk.shape[0])),flush=True)
                    
                    asr = asr/(idx_atk.shape[0])/self.args.attack_num
                    # asr = asr/(idx_atk.shape[0])
                    # asr = asr/(idx_atk.shape[0])/(labels.max().item()+1)
                    print("Defense: {},Test_model: {},Overall ASR: {:.4f}".format(args.defense_mode,args.test_model,asr), flush=True)
                    
        print('==test over==', flush=True)
        
    def new_test(self,idx_train,idx_eval,idx_atk,data,labels,idx_clean_test,mask_edge_index):
        loss_labels = calculate_weights(idx_train,labels)
        loss_labels = torch.tensor(loss_labels).to(self.device)

        args = self.args
        device =self.device
        
        
        file_path = './test/{}/origin-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        data_dict = torch.load(file_path) 
        poison_x = data_dict['poison_x']  
        poison_edge_index = data_dict['poison_edge_index']  
        poison_edge_weights = data_dict['poison_edge_weights']  
        poison_labels = data_dict['poison_labels']  
        idx_attach = data_dict['idx_attach']
        
        origin_poison_x = poison_x
        origin_poison_edge_index = poison_edge_index
        origin_poison_edge_weights = poison_edge_weights
        origin_poison_labels = poison_labels
        origin_idx_attach = idx_attach
        
        
        file_path = './test/{}/fit-{}-{}-{}-{}.pth'.format(args.dataset,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
        dataFit_dict = torch.load(file_path) 
        poison_x_list = dataFit_dict['poison_x_list']  
        poison_edge_list = dataFit_dict['poison_edge_list']  
        poison_weights_list = dataFit_dict['poison_weights_list'] 
        poison_labels_list = dataFit_dict['poison_labels_list'] 
        
        
        rs = np.random.RandomState(args.seed)
        seeds = rs.randint(1000,size=15)
        seeds = list(seeds)
        seeds.insert(0, args.seed) 
        # models = ['GNNGuard','RobustGCN']
        models = ['GCN','GAT', 'GraphSage']
        # defense_modes = ['none','prune','isolate']
        defense_modes = ['reconstruct']
        
        
        
        for seed in seeds:
            
            args.seed = seed
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

            for defense_mode in defense_modes:
                
                args.defense_mode = defense_mode
                
                idx_attach = origin_idx_attach
                
                bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(self.device)
                if(self.args.defense_mode == 'prune'):
                    poison_edge_index,poison_edge_weights = prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,self.device,large_graph=False)
                elif(args.defense_mode == 'isolate'):
                    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,device,large_graph=False)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
                    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
                    idx_attach = torch.LongTensor(list(set(idx_attach) - set(rel_nodes))).to(device)
                elif(args.defense_mode == 'reconstruct'):
                    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(self.args,origin_poison_edge_index,origin_poison_edge_weights,poison_x,data.x,data.edge_index,self.device, idx_attach, large_graph=True)
                    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
                
                if self.args.fit_attach_num<=len(idx_attach):
                    new_idx_attach = idx_attach[:self.args.fit_attach_num]
                else:
                    new_idx_attach = idx_attach
                    
                
                for test_model in models:
                    args.test_model = test_model
                    asr = 0
                
                    
                    test_model = model_construct(args,args.test_model,data,device).to(device)
                    
                    
                    test_model.new_fit(poison_x_list, poison_edge_list, poison_weights_list, poison_labels_list, [idx_train,new_idx_attach], idx_eval, None,train_iters=300, verbose=False)
                    
                        
                    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
                    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
                    # poison_edge_index,poison_x,poison_edge_weights= model.test(idx_atk,idx_attach,features,edge_index)
                    clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

                    print("accuracy on clean test nodes: {:.4f}".format(clean_acc), flush=True)
                    
            
                    
                    for index in range(labels.max().item()+1):
                       
                        file_path ='./test/{}/label{}-{}-{}-{}-{}.pth'.format(args.dataset,index,args.num_prompts,args.test_thr,args.fit_attach_num,args.trojan_epochs)
                        dataFit_dict = torch.load(file_path) 
                        induct_x_list = dataFit_dict['induct_x_list']  
                        induct_edge_index_list = dataFit_dict['induct_edge_index_list']  
                        induct_edge_weights_list = dataFit_dict['induct_edge_weights_list'] 
                        trojan_edge_index_list = dataFit_dict['trojan_edge_index_list'] 
                        target_class_list = dataFit_dict['target_class_list'] 
                        relabeled_node_idx_list = dataFit_dict['relabeled_node_idx_list'] 
                        
                        
                        for induct_x,induct_edge_index,induct_edge_weights,trojan_edge_index,target_class,relabeled_node_idx in zip(induct_x_list,induct_edge_index_list,induct_edge_weights_list,trojan_edge_index_list,target_class_list,relabeled_node_idx_list):
                           
                            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                        
                            #
                            if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                                origin_edge_index = induct_edge_index
                                induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
                                normal_edges_count,special_edges_count = count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index)
                                # count_removed_edges(origin_edge_index, induct_edge_index, trojan_edge_index, origin_edge_index)
                                # normal_edges_count,special_edges_count = count_edges(old_edge_index,induct_edge_index,total_nodes)
                                # total_special_edges_count += special_edges_count
                                # total_normal_edges_count += normal_edges_count
                                
                            
                            output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                            train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==target_class).float().mean()
                            asr+=train_attach_rate
                    asr = asr/(labels.max().item()+1)/len(idx_atk)
                    print("Defense: {},Test_model: {},Overall ASR: {:.4f}".format(args.defense_mode,args.test_model,asr), flush=True)
                    