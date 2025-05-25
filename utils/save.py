import os
import json
import pickle
import pandas as pd
from dataset import GraphBuilder


def get_events_and_labels(dataname, event_path, label_path):
    events, labels = [], []
    onehot_map = {'0': [1, 0, 0, 0], '1': [0, 1, 0, 0], '2': [0, 0, 1, 0], '3': [0, 0, 0, 1]} \
        if dataname == 'weibo' else {'0': [1, 0], '1': [0, 1], '2': [1, 0], '3': [0, 1]}

    label_dic = {}
    with open(label_path, 'r', encoding='utf-8') as file:
        df = pd.read_csv(file, header=0, dtype={'eid': str, 'label': str})
        for row in range(len(df)):
            event = df.loc[row, 'eid']
            label = df.loc[row, 'label']
            label_dic[event] = onehot_map[label]
        file.close()
    
    with open(event_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            event = line.strip()
            label = label_dic[event]
            events.append(event)
            labels.append(label)
        file.close()
        
    return events, labels

def save_data(dataname):
    data_path = os.path.join(os.getcwd(), 'data', dataname)
    content_path = os.path.join(data_path, 'content.json')
    contents = None
    with open(content_path, 'r', encoding='utf-8') as file:
        contents = json.load(file)
        file.close()
        
    label_path = os.path.join(data_path, 'label.csv')
    train_event_path = os.path.join(data_path, 'train', 'events.txt')
    dev_event_path = os.path.join(data_path, 'dev', 'events.txt')
    test_event_path = os.path.join(data_path, 'test', 'events.txt')

    train_events, train_labels = get_events_and_labels(dataname, train_event_path, label_path)
    dev_events, dev_labels = get_events_and_labels(dataname, dev_event_path, label_path)
    test_events, test_labels = get_events_and_labels(dataname, test_event_path, label_path)
    
    train_graphs, train_macro_graphs = GraphBuilder(contents, train_events, train_labels).build()
    dev_graphs, dev_macro_graphs = GraphBuilder(contents, dev_events, dev_labels).build()
    test_graphs, test_macro_graphs = GraphBuilder(contents, test_events, test_labels).build()
    
    with open(os.path.join(data_path, 'train', 'graphs.pkl'), 'wb') as file:
        data = {'graphs': train_graphs, 'macro_graphs': train_macro_graphs, 'events': train_events, 'labels': train_labels}
        pickle.dump(data, file)
        file.close()

    with open(os.path.join(data_path, 'dev', 'graphs.pkl'), 'wb') as file:
        data = {'graphs': dev_graphs, 'macro_graphs': dev_macro_graphs, 'events': dev_events, 'labels': dev_labels}
        pickle.dump(data, file)
        file.close()

    with open(os.path.join(data_path, 'test', 'graphs.pkl'), 'wb') as file:
        data = {'graphs': test_graphs, 'macro_graphs': test_macro_graphs, 'events': test_events, 'labels': test_labels}
        pickle.dump(data, file)
        file.close()

     
def cal_num(data_path):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
        print(len(data['events']))
        file.close()
    
    
if __name__ == '__main__':
    save_data('pheme')
    # dataname = 'pheme'
    # mode = 'train'
    # data_path = os.path.join(os.getcwd(), 'data', dataname, mode, 'graphs.pkl')
    # cal_num(data_path)
    