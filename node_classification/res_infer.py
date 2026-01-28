import torch
from collections import Counter
import re
from sklearn.metrics import f1_score, accuracy_score
import numpy as np



graph_name = 'citeseer'

ratio = 0.3

relax = True
if relax:
    tn_class_results = torch.load(f'./{graph_name}/tn_class_results_{ratio}_rel.pt')
else:
    tn_class_results = torch.load(f'./{graph_name}/tn_class_results_{ratio}.pt')


PATH_data = '../dataset'
data = torch.load(f'{PATH_data}/{graph_name}.pt', weights_only=False)

nodes = data['test_mask'].nonzero().squeeze().tolist()
labels_whole = data['label_whol']
y = data.y

print(len(tn_class_results))

# assert len(set(nodes) & set(tn_class_results.keys())) == len(nodes) == len(tn_class_results)

def find_most_frequent_label(labels_whole, long_string):

    lower_labels = []
    for label in labels_whole:
        lower_label = label.lower()
        lower_labels.append(lower_label)

            
    lower_long_string = long_string.lower()
    
 
    label_counts = Counter()
    

    for lower_label in lower_labels:

        count = lower_long_string.count(lower_label)
        
        if count > 0:
            label_counts[lower_label] = count

    if not label_counts:
        return (None, 0)
    

    most_common_lower_label, max_count = label_counts.most_common(1)[0]

    return (most_common_lower_label, max_count)



if graph_name != 'book':
    lower_labels = []
    for label in labels_whole:
        lower_label = label.lower()
        lower_labels.append(lower_label)

    preds = []
    gt = []
    for node, preds_string in tn_class_results.items():
        label, count = find_most_frequent_label(labels_whole=labels_whole, long_string=preds_string)
        if label in lower_labels:
            index = lower_labels.index(label)
        else: index = -1
        preds.append(index)
        gt.append(y[node].item())



    acc = accuracy_score(np.array(gt), np.array(preds))
    f1_weighted = f1_score(np.array(gt), np.array(preds), average='weighted')
    print(acc)
    print(f1_weighted)

else:
    node2label = dict()
    for i in range(y.shape[0]):
        labels = torch.where(y[i] == 1)[0].tolist()
        node2label[i] = labels
    gt = []
    lower_labels = []
    for label in labels_whole:
        lower_label = label.lower()
        lower_labels.append(lower_label)

    preds = []
    for node, preds_string in tn_class_results.items():
        label, count = find_most_frequent_label(labels_whole=labels_whole, long_string=preds_string)
        if label in lower_labels:
            index = lower_labels.index(label)
        else: index = -1
        preds.append(index)
        gt.append(node2label[node])
    count = 0
    for pred, gtlist in zip(preds, gt):
        if pred in gtlist:
            count += 1
    print(count/len(preds))