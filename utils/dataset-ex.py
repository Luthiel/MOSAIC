import os
import json
import torch
import numpy as np

from math import exp
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from models.text.encoder import MPNet
from utils.tree import construct_tree
from sklearn.cluster import AgglomerativeClustering
from PIL import Image
from models.graph.upgrade import DUM
from torchvision import transforms as T
from transformers import AutoTokenizer


# -------------------------------- Macro Graph Reconstruct ----------------------------------
def find_descendants_iterative(node):
    """迭代获取节点的所有后代索引，避免递归栈溢出"""
    # 初始化后代列表和栈
    descendants = []
    stack = [node]
    # 当栈不为空时，循环执行
    while stack:
        # 弹出栈顶节点
        current = stack.pop()
        descendants.append(current.idx)
        stack.extend(current.children)  # 将所有子节点加入栈
    return descendants

def cluster_nodes(features, distance_threshold=1.0, linkage='ward'):
    """对节点特征进行层次聚类"""
    cluster = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold, 
        linkage=linkage
    )
    
    if len(features) == 1:
        return [0]
    return cluster.fit_predict(features)

def aggregate_cluster_features(indices, x, timestamps, depth_value):
    """聚合聚类后的特征、时间戳和深度"""
    indices = np.array(indices)  # 确保是 numpy 数组以支持高效索引
    return (
        np.mean(x[indices], axis=0),   # 批量计算特征均值
        np.mean(timestamps[indices]),  # 批量计算时间戳均值
        depth_value                    # 深度值
    )

def reconstruct_macro(x, edge_index, timestamps, depths, event_id):
    # print("# -------------------------")
    # print(x.shape)
    x = x.detach().numpy()
    edge_index = edge_index.detach().numpy()
    timestamps = timestamps.detach().numpy()
    depths = depths.detach().numpy()

    new_x = []
    new_edge_index = [[], []]
    new_timestamps = []
    new_depths = []

    # 构建树并保留根节点
    tree = construct_tree(x, edge_index, timestamps, depths, event_id)
    root = tree[0]
    new_x.append(root.x)
    new_timestamps.append(root.timestamp)
    new_depths.append(int(root.depth))

    # 次顶层节点聚类 (depth=1)
    subtop_indices = np.where(depths == 1)[0] 
    if len(subtop_indices) == 0:
        return new_x, new_edge_index, new_timestamps, new_depths

    subtop_feats = x[subtop_indices]
    # print(f'subtop feats shape are {subtop_feats.shape}')
    subtop_labels = cluster_nodes(subtop_feats)
    # print(f'subtop labels are {subtop_labels}')
    subtop_cluster = {}
    for node_idx, label in zip(subtop_indices, subtop_labels):
        subtop_cluster.setdefault(label, []).append(node_idx)

    # 添加次顶层聚类结果
    for label, nodes in subtop_cluster.items():
        feat, ts, depth = aggregate_cluster_features(nodes, x, timestamps, 1)
        new_x.append(feat)
        new_edge_index[0].append(0)         
        new_edge_index[1].append(label + 1) 
        new_timestamps.append(ts)
        new_depths.append(depth)

    next_index = len(subtop_cluster) + 1
    subtop_label_map = {idx: label + 1 for idx, label in zip(subtop_indices, subtop_labels)}
    
    edge_parents = set(edge_index[0])

    for subtop_idx in subtop_indices:
        if subtop_idx not in edge_parents:  # 跳过无子节点的次顶层节点
            continue
        descendants = find_descendants_iterative(tree[subtop_idx])
        if not descendants:
            continue

        desc_feats = x[descendants]
        desc_labels = cluster_nodes(desc_feats)
        desc_cluster = {}
        for desc_idx, label in zip(descendants, desc_labels):
            desc_cluster.setdefault(label, []).append(desc_idx)

        # 添加子树聚类结果
        for label, nodes in desc_cluster.items():
            feat, ts, depth = aggregate_cluster_features(nodes, x, timestamps, 2)
            new_x.append(feat)
            new_edge_index[0].append(subtop_label_map[subtop_idx])  
            new_edge_index[1].append(next_index + label)            
            new_timestamps.append(ts)
            new_depths.append(depth)

        next_index += len(desc_cluster)
    
    new_x = np.array(new_x)
    new_edge_index = np.array(new_edge_index)
    new_timestamps = np.array(new_timestamps)
    new_depths = np.array(new_depths)
    
    return torch.tensor(new_x), torch.tensor(new_edge_index), torch.tensor(new_timestamps), torch.tensor(new_depths)

class GraphBuilder:
    def __init__(self, contents, events, img_post):
        self.encoder = MPNet().eval()
                
        self.contents = contents
        self.events = events
        self.img_post = img_post
        self.onehot_map = {'0': [1, 0, 0, 0], '1': [0, 1, 0, 0], '2': [0, 0, 1, 0], '3': [0, 0, 0, 1]}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.denoise = DUM(384, 256, 384).to(self.device)

    def _get_edge_weight(self, node_feats, edges, times, depths,event_id, alpha=.5, beta=1., scale=1., use_time_bias=True):
        node_num = len(node_feats)
        
        adj_mask = torch.zeros((node_num, node_num)).to(node_feats.device)
        decay_factor = torch.zeros((node_num, node_num)).to(node_feats.device)
        max_time, min_time = max(times), min(times)
        
        # print(f'node num is {node_num}')
        # print(f'edges 0 is {len(edges[0])}, edges 1 is {len(edges[1])}')
        # print(f'max edge 0 is {max(edges[0])}, max edge 1 is {max(edges[1])}')
        for row, col in zip(edges[0], edges[1]):
            try:
                adj_mask[row][col] = 1
            except:
                print(f'adj error in {event_id}')
                break
            delta_time = float((times[col] - min_time) / (max_time - min_time) * scale)
            if use_time_bias:
                decay_factor[row][col] = exp(-alpha * depths[row]) + self._get_time_bias(delta_time)
            else:
                decay_factor[row][col] = exp(-alpha * depths[row] - beta * delta_time)
            
        sim_matrix = torch.matmul(node_feats, node_feats.t())
        edge_weights = sim_matrix * adj_mask * decay_factor
        
        weight_lst = []
        for row, col in zip(edges[0], edges[1]):
            weight_lst.append(edge_weights[row][col])
        
        return weight_lst
    
    def _get_time_bias(self, time_diffs):
        epsilon = 1
        gamma = 0.3
        delta = 0.5
        return -epsilon * max(0, time_diffs) + gamma * exp(-(time_diffs * time_diffs) / (2 * delta ** 2))
    
    """
    def _check_smooth(self, edge_index):
        
        sorted(edge_index, key=lambda x: x[1])
        kids = edge_index[1]
        if max(kids) > len(kids):
            print('under smoothing!')
            boundary = len(kids)
            map = {}
            for i, kid in enumerate(kids):
                if kid > boundary:
                    map[kid] = i + 1
            
            for i, (parent, kid) in enumerate(zip(edge_index[0], edge_index[1])):
                if parent in map.keys():
                    edge_index[0][i] = map[parent]
                if kid in map.keys():
                    edge_index[1][i] = map[kid]
     
        return edge_index
    """
    
    def build(self):
        graphs, macro_graphs = [], []
        for event_id in self.events:
            torch.cuda.empty_cache()
            conversation, temporals, depths, edge_index = [], [], [], [[], []]
            tree = self.contents[event_id]
            for node, info in tree.items():
                if node == '0':
                    conversation.append(info['content'])
                    temporals.append(info['timestamp'])
                    depths.append(info['depth'])
                    continue
                
                conversation.append(info['content'])
                edge_index[0].append(int(info['parent_id']))
                edge_index[1].append(int(node))
                temporals.append(info['timestamp'])
                depths.append(info['depth'])

            temporals = torch.tensor(temporals, dtype=torch.long)
            with torch.no_grad():
                node_features_cuda = self.encoder(conversation, temporals)
            # print(len(node_features_cuda))
            node_features = node_features_cuda.cpu()

            del node_features_cuda
            torch.cuda.empty_cache()
            
            # edge_index = self._check_smooth(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
            depths = torch.tensor(depths, dtype=torch.long).view(-1, 1)
            edge_weights = self._get_edge_weight(node_features, edge_index, temporals, depths, event_id)
            
            edge_weights = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
            # print("# ------------------------------------")
            # print(f'x shape before DUM are {node_features.shape}')
            x_cuda, edge_weights_cuda = self.denoise(node_features, edge_index, edge_weights)

            x = x_cuda.cpu()
            edge_weights = edge_weights_cuda.cpu()
            
            del x_cuda, edge_weights_cuda
            torch.cuda.empty_cache()
            
            raw_label = self.img_post[event_id]['label']
            label = self.onehot_map[raw_label]
            
            graphs.append(Data(x=x, 
                               edge_index=edge_index, 
                               edge_attr=edge_weights, 
                               depths=depths, 
                               timestamps=temporals,
                               y=label))
            
            macro_x, macro_edge, macro_times, macro_depths = reconstruct_macro(x, edge_index, temporals, depths, event_id)
            macro_graphs.append(Data(x=macro_x,
                            edge_index=macro_edge, 
                            edge_attr=edge_weights, 
                            depths=macro_depths, 
                            timestamps=macro_times,
                            y=label))
            
            del node_features, x, edge_weights, macro_x, macro_edge, macro_times, macro_depths
            torch.cuda.empty_cache()
        
        return graphs, macro_graphs

class GraphDataset(Dataset):
    def __init__(self, contents, events, img_post):
        self.onehot_map = {'0': [1, 0, 0, 0], '1': [0, 1, 0, 0], '2': [0, 0, 1, 0], '3': [0, 0, 0, 1]}
        self.graphs, self.macro_graphs = GraphBuilder(contents, events, img_post).build()
        self.img_post = img_post
        self.events = events
        
    def __getitem__(self, idx):
        
        graph = self.graphs[idx]
        macro_graph = self.macro_graphs[idx]

        event_id = self.events[idx] # 注意在 build 的时候，idx 和 event_id 是一一对应的，即按照events列表的顺序添加
        img = self.img_post[event_id]['img_no']
        post = self.img_post[event_id]['post']
        label = self.onehot_map[self.img_post[event_id]['label']]
        return graph, macro_graph, img, post, label
        
    def __len__(self):
        return len(self.events)

def get_events(event_path):
    with open(event_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        events = []
        for line in lines:
            item = line.strip().split('\t')
            events.append(item[0])
        file.close()
    return events

def mosaic_collate_fn(batch):
    
    processed_batch = []
    for item in batch:
        graph, macro_graph, img, post, label = item
        transform = T.Compose([
            T.Resize((336, 336)),
            T.ToTensor()])
        pixel = transform(img)
        print("in dataloader, pixels type  is", type(pixel))
        processed_batch.append((graph, macro_graph, pixel, post, label))

    if isinstance(batch[0], tuple):  
        graphs = [item[0] for item in processed_batch]
        macro_graphs = [item[1] for item in processed_batch]
        images = torch.stack([item[2] for item in processed_batch])
        texts = [item[3] for item in batch]  
        labels = torch.tensor([item[4] for item in batch])
        return graphs, macro_graphs, images, texts, labels
    else:
        return torch.stack(processed_batch)

def load_data(batch_size, dataname):
    data_path = os.path.join(os.getcwd(), 'data', dataname)
    content_path = os.path.join(data_path, 'content.json')
    contents = None
    with open(content_path, 'r', encoding='utf-8') as file:
        contents = json.load(file)
        file.close()
    
    img_path = os.path.join(data_path, 'images')
    post_path = os.path.join(data_path, 'content.txt')
    
    img_lst = os.listdir(img_path)
    img_map = {}
    for img in img_lst:
        img_id = img.split('.')[0]
        img_map[img_id] = Image.open(os.path.join(img_path, img)).convert('RGB')
        
    img_post = {}
    with open(post_path, 'r', encoding='utf-8') as file:
        line = file.readlines()
        for line in line:
            eid, img_no, post, label = line.strip().split('\t')
            if not eid in img_post.keys():
                img_post[eid] = {}
            img_post[eid]['post'] = post
            img_post[eid]['img_no'] = img_map[img_no]
            img_post[eid]['label'] = label
        file.close()
    
    train_path = os.path.join(data_path, 'train', 'events.txt')
    dev_path = os.path.join(data_path, 'dev', 'events.txt')
    test_path = os.path.join(data_path, 'test', 'events.txt')

    # train_events, train_imgs, train_posts, train_labels = read_graph_data(train_path, img_post)
    # dev_events, dev_imgs, dev_posts, dev_labels = read_graph_data(dev_path, img_post)
    test_events = get_events(test_path)
    
    assert contents is not None, 'Contents loading failed! Please recheck the content file path!'
    # train_data = GraphDataset(contents, train_events, train_imgs, train_posts, train_labels)
    # dev_data = GraphDataset(contents, dev_events, dev_imgs, dev_posts, dev_labels)
    test_data = GraphDataset(contents, test_events, img_post)
    
    # train_iter = DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True, collate_fn=mosaic_collate_fn)
    # dev_iter = DataLoader(dev_data, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True, collate_fn=mosaic_collate_fn)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True, collate_fn=mosaic_collate_fn)

    train_iter = None
    dev_iter = None

    return train_iter, dev_iter, test_iter

# -------------------------------- Vision Text Dataset ----------------------------------

class VisionTextDataset(Dataset):
    def __init__(self, data, events):
        self.onehot_map = {'0': [1, 0, 0, 0], '1': [0, 1, 0, 0], '2': [0, 0, 1, 0], '3': [0, 0, 0, 1]}
        self.data = data
        self.events = events

    def __getitem__(self, idx):
        event_id = self.events[idx]

        img = self.data[event_id]['img_no']
        post = self.data[event_id]['post']
        
        label = self.onehot_map[self.data[event_id]['label']]
        return img, post, label
        
    def __len__(self):
        return len(self.events)
    

def mvp_collate_fn(batch):
    # 假设 batch 是元组或列表，检查每个元素类型并处理
    processed_batch = []
    for item in batch:
        img, text, label = item
        transform = T.Compose([
            T.Resize((336, 336)),
            T.ToTensor()])
        processed_item = transform(img)
        processed_batch.append((processed_item, text, label))

    # 根据 batch 的结构返回
    if isinstance(batch[0], tuple):  # 如果是 (image, text) 的形式
        images = torch.stack([item[0] for item in processed_batch])
        texts = [item[1] for item in batch]  # 文本保持列表
        labels = torch.tensor([item[2] for item in batch])  # 标签保持列表
        return images, texts, labels
    else:
        return torch.stack(processed_batch)

def load_vision_text(dataname, batch_size):
    root_dir = os.path.join(os.getcwd(), 'data') # root/Mosaic
    data_path = os.path.join(root_dir, dataname)
    img_path = os.path.join(data_path, 'images')
    post_path = os.path.join(data_path, 'content.txt')
    
    img_lst = os.listdir(img_path)
    img_map = {}
    for img in img_lst:
        img_id = img.split('.')[0]
        img_map[img_id] = Image.open(os.path.join(img_path, img)).convert('RGB')
        
    data = {}
    with open(post_path, 'r', encoding='utf-8') as file:
        line = file.readlines()
        for line in line:
            eid, img_no, post, label = line.strip().split('\t')
            if not eid in data.keys():
                data[eid] = {}
            data[eid]['post'] = post
            data[eid]['img_no'] = img_map[img_no]
            data[eid]['label'] = label
        file.close()
        
    train_path = os.path.join(data_path, 'train', 'events.txt')
    dev_path = os.path.join(data_path, 'dev', 'events.txt')
    test_path = os.path.join(data_path, 'test', 'events.txt')

    train_events = get_events(train_path)
    dev_events = get_events(dev_path)
    test_events = get_events(test_path)
    
    train_data = VisionTextDataset(data, train_events)
    dev_data = VisionTextDataset(data, dev_events)
    test_data = VisionTextDataset(data, test_events)
    
    train_iter = DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=mvp_collate_fn)
    dev_iter = DataLoader(dev_data, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=mvp_collate_fn)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=mvp_collate_fn)

    return train_iter, dev_iter, test_iter

'''
def save(contents, events, labels, dataname, save_path, use_time_enc):
    print(f'total events: {len(events)}')
    max_size = 200 if dataname == 'weibo' else 50
    x_batch, index_batch, weights_batch, depth_batch, label_batch, dfs_batch, bfs_batch = \
            GraphBuilder(contents, events, labels, dataname, use_time_enc).build_batch_graphs(events, max_size)
          
    obj = {'graphs': x_batch, 
           'edge_index': index_batch, 
           'edge_weights': weights_batch, 
           'depth': depth_batch, 
           'label': label_batch, 
           'dfs': dfs_batch,
           'bfs': bfs_batch,
           }  
    
    with open(save_path, 'wb') as file:
        pickle.dump(obj, file)
        
    end = time()
    print(f"save {dataname} data done! Cost time: {end - start:.2f}s")

if __name__ == "__main__":
    dataname = 'pheme'
    cur_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_path = os.path.join(cur_dir, 'train', dataname + '.txt')
    dev_path = os.path.join(cur_dir, 'dev', dataname + '.txt')
    test_path = os.path.join(cur_dir, 'test', dataname + '.txt')

    train_events, train_labels = read_data(train_path)
    dev_events, dev_labels = read_data(dev_path)
    test_events, test_labels = read_data(test_path)
    
    with open(os.path.join(cur_dir, dataname + '_new_content_2.json'), 'r', encoding='utf-8') as file:
        contents = json.load(file)
        
    save(contents, train_events, train_labels, dataname, save_path=os.path.join(cur_dir, 'train', dataname + '_features_without_time.pkl'), use_time_enc=False)
    save(contents, dev_events, dev_labels, dataname, save_path=os.path.join(cur_dir, 'dev', dataname + '_features_without_time.pkl'), use_time_enc=False)
    save(contents, test_events, test_labels, dataname, save_path=os.path.join(cur_dir, 'test', dataname + '_features_without_time.pkl'), use_time_enc=False)
'''