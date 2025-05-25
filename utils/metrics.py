from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(preds, labels):
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    macro_f1 = f1_score(labels, preds, average='macro')
    return acc, precision, recall, f1, macro_f1