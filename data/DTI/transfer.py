import csv
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import pandas as pd

G = nx.Graph()

with open('entity_list.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    column = [row['Entity_ID'] for row in reader]
    nodes = tuple(column)
    G.add_node(nodes)

folds = ['fold1','fold2', 'fold3', 'fold4', 'fold5']
for i in folds:
    with open(i + '.csv','r') as f:
        edgess = csv.reader(f)

        for edges in edgess:
            edge = tuple(edges)
            G.add_nodes_from(edge)
     # g: nodes, edges

    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=10, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.wv.save_word2vec_format('dti'+ i +'.csv')

# folds = ['fold1','fold2', 'fold3', 'fold4', 'fold5']
# for i in folds:
#     with open('./'+ i + '/train.csv','r') as f:
#         edgess = csv.reader(f)
#         # edgess = pd.read_csv(f, index_col=0)
#         for edges in edgess:
#             edge = tuple(edges)
#             print(edge)
#             # break


# folds = ['fold1','fold2', 'fold3', 'fold4', 'fold5']
# for i in folds:
#     with open('./'+ i + '/train.csv','r') as f:
#         data = pd.read_csv(f, index_col=0)
#         d = data.to_csv( i + '.csv', index=False)
