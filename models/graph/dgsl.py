import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch, Data
from models.graph.mamba import Mamba2Block
from torch_scatter import scatter
from models.graph.upgrade import DUM

class DynamicGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2):
        super(DynamicGCN, self).__init__()
        self.in_dim = in_dim
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.num_layers = num_layers
        self.match = nn.Linear(384, in_dim) # to fit the dimension of vision-text features
    
    def forward(self, x, edge_index, edge_weight=None):
        
        if not x.shape[-1] == self.in_dim:
            x = self.match(x)
            
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x
    
class DGSL(nn.Module):
    def __init__(self, in_dim, hidden_dim, state_dim, need_macro=True, use_dum=True, nodes_per_group=5, pooling_method='last'):
        super(DGSL, self).__init__()
        self.in_dim = in_dim
        
        self.denoise = DUM(384, 256, 384)
        
        self.micro_gnn = DynamicGCN(in_dim, hidden_dim, 1)
        self.macro_gnn = DynamicGCN(in_dim, hidden_dim, 1)

        self.mamba = nn.ModuleList()
        for _ in range(len(self.micro_gnn.layers)):
            self.mamba.append(Mamba2Block(hidden_dim, state_dim))
            
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim) if need_macro else nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2), 
        )
        
        self.hidden_dim = hidden_dim
        self.nodes_per_group = nodes_per_group
        self.pooling_method = pooling_method
        self.need_macro = need_macro
        self.use_dum = use_dum
    
    def preprocess_batch(self, batch_data, target_seq_len=50):
        batch_size = len(batch_data)
        if batch_size == 0:
            raise ValueError("batch_data is empty")
        
        device = batch_data[0].x.device
        x_tensor = torch.zeros(batch_size, target_seq_len, self.nodes_per_group, 384, device=device) # 384 is the original dimension of node features
        timestamps_tensor = torch.zeros(batch_size, target_seq_len, device=device)
        mask_tensor = torch.zeros(batch_size, target_seq_len, dtype=torch.bool, device=device)
        
        data_list = []
        seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b, data in enumerate(batch_data):
            if not hasattr(data, 'x') or data.x.size(0) == 0:
                print(f"Warning: Graph {b} has no nodes, generating empty snapshots")
                for i in range(target_seq_len):
                    data_list.append(Data(
                        x=torch.zeros((0, 384), device=device),
                        edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                        edge_attr=torch.empty((0,), device=device),
                        graph_idx=b,
                        time_idx=i
                    ))
                seq_lengths[b] = 0
                continue
            
            timestamps = data.timestamps if hasattr(data, 'timestamps') else torch.arange(data.x.size(0))
            if timestamps.size(0) != data.x.size(0):
                raise ValueError(f"Graph {b}: timestamps length mismatch with nodes")
            
            sorted_idx = torch.argsort(timestamps)
            x = data.x[sorted_idx]
            timestamps = timestamps[sorted_idx]
            edge_index = self.remap_edge_index(data.edge_index, sorted_idx) if hasattr(data, 'edge_index') else torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else torch.ones(edge_index.size(1), device=device)
            # print(f'group edge weight before mask: {edge_weight.shape}')
            
            num_nodes = x.size(0)
            interval = max(num_nodes // target_seq_len, self.nodes_per_group) if num_nodes >= target_seq_len else self.nodes_per_group
            num_snapshots = min((num_nodes + interval - 1) // interval, target_seq_len)
            seq_lengths[b] = num_snapshots
            padding = target_seq_len - num_snapshots
            
            # print(f"Graph {b}: num_nodes={num_nodes}, interval={interval}, num_snapshots={num_snapshots}, padding={padding}")

            for i in range(padding):
                # print(f"Graph {b}: Adding padding snapshot time_idx={i}")
                data_list.append(Data(
                    x=torch.zeros((0, 384), device=device),
                    edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                    edge_attr=torch.empty((0,), device=device),
                    graph_idx=b,
                    time_idx=i
                ))

            for i in range(num_snapshots):
                end_idx = min((i + 1) * interval, num_nodes)
                group_x = x[:end_idx]
                x_tensor[b, i + padding, :min(end_idx, self.nodes_per_group)] = \
                    group_x[-self.nodes_per_group:] if end_idx > self.nodes_per_group else group_x
                timestamps_tensor[b, i + padding] = timestamps[end_idx - 1]
                mask_tensor[b, i + padding] = True # 只有真实的 snapshot 才需要 mask 为 True，因为是前向 pad

                mask = (edge_index[0] < end_idx) & (edge_index[1] < end_idx)
                group_edge_index = edge_index[:, mask]
                group_edge_weight = edge_weight[mask]
                
                data_list.append(Data(
                    x=group_x,
                    edge_index=group_edge_index,
                    edge_attr=group_edge_weight,
                    graph_idx=b,
                    time_idx=i + padding
                ))

        batch = Batch.from_data_list(data_list)
            
        return x_tensor, timestamps_tensor, mask_tensor, batch, seq_lengths
    
    def remap_edge_index(self, edge_index, sorted_idx):
        num_nodes = int(edge_index.max()) + 1
        idx_map = -torch.ones(num_nodes, dtype=torch.long, device=edge_index.device)
        idx_map[sorted_idx] = torch.arange(len(sorted_idx), device=edge_index.device)
        
        remapped = idx_map[edge_index]
        
        # 检查映射是否合法
        if (remapped < 0).any():
            print("注意：edge_index 包含不在 sorted_idx 内的节点")
        
        return remapped

    def pool_sequence(self, mamba_output, seq_lengths, mask_tensor):
        batch_size, max_seq_len = mamba_output.shape[:2]
        
        if self.pooling_method == 'mean':
            masked_output = mamba_output * mask_tensor.unsqueeze(-1) # [64, 50, 256]
            return masked_output.sum(dim=1) / (seq_lengths.unsqueeze(-1) * self.nodes_per_group)
        elif self.pooling_method == 'max':
            valid_output = [mamba_output[b, :seq_lengths[b]] for b in range(batch_size)]
            return torch.stack([out.max(dim=0)[0].max(dim=0)[0] for out in valid_output])
        elif self.pooling_method == 'last':
            last_indices = (seq_lengths - 1).clamp(min=0)
            return torch.stack([mamba_output[b, last_indices[b]].mean(dim=0) for b in range(batch_size)])
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def forward(self, macro_graphs, batch_data):
        # seq_length 记录的是真实的 snapshot 数量
        x_tensor, timestamps_tensor, mask_tensor, batch, seq_lengths = self.preprocess_batch(batch_data)
        batch_size, max_seq_len, max_nodes = x_tensor.shape[:3]
        
        # 批量 GCN 计算
        micro_x, micro_weight, edge_index = batch.x, batch.edge_attr, batch.edge_index
        if self.use_dum:
            micro_x, micro_weight = self.denoise(micro_x, edge_index, micro_weight)
            
        gcn_output = self.micro_gnn(micro_x, edge_index, micro_weight)
        
        # 重塑回时间序列格式
        embeddings = torch.zeros(batch_size, max_seq_len, max_nodes, self.hidden_dim, device=x_tensor.device)
        for b in range(batch_size):
            for t in range(seq_lengths[b]):
                # 访问 batch 中的 batch id 和 time id
                mask = (batch.graph_idx.squeeze() == b) & (batch.time_idx.squeeze() == t)
                if mask.sum() == 0:
                    continue
                node_indices = torch.where(mask)[0]
                num_nodes = min(node_indices.size(0), max_nodes)
                
                index = node_indices[:num_nodes]
                if index < gcn_output.size(0): # pheme-train 个别数据有问题，没找出来，所以直接截断
                    embeddings[b, t, :num_nodes] = gcn_output[index]
        
        out_mamba = torch.mean(embeddings, dim=-2)
        for i in range(len(self.mamba)):
            out_mamba = self.mamba[i](out_mamba)
        
        # 池化并分类, out_mamba.shape = [batch_size, seq_len, hidden_dim], default [64, 50, 256]
        micro_pooled = out_mamba[:, -1].squeeze(1)
        
        if self.need_macro:
            batch_macro = Batch.from_data_list(macro_graphs)
            macro_feat = self.macro_gnn(batch_macro.x, batch_macro.edge_index, batch_macro.edge_attr)

            # 重塑为 [batch_size, hidden_dim]
            node_to_graph = batch_macro.batch
            num_graphs = batch_macro.num_graphs
            num_nodes_per_graph = scatter(torch.ones_like(node_to_graph), node_to_graph, dim=0, reduce='sum')
            max_nodes = num_nodes_per_graph.max().item()
            
            feature_dim = macro_feat.size(-1)  
            stacked_macro_feat = torch.zeros(num_graphs, max_nodes, feature_dim, device=macro_feat.device)

            # 按图分配特征
            for i in range(num_graphs):
                mask = (node_to_graph == i)  
                graph_feat = macro_feat[mask]  
                stacked_macro_feat[i, :graph_feat.size(0)] = graph_feat 
            
            macro_pooled = stacked_macro_feat.mean(dim=1)
            pooled = torch.cat([macro_pooled, micro_pooled], dim=-1)
        
            graph_feats = self.mlp(pooled) # [batch_size, hidden_dim]
        else:
            pooled = micro_pooled
            graph_feats = self.mlp(pooled)
            
        return graph_feats