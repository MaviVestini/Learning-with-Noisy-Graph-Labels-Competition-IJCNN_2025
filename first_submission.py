import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,  
    global_mean_pool, 
)
import numpy as np



# Model definition

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        
        self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual)
        self.pool = global_mean_pool

        self.graph_pred_linear1 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data,train=False):
        h_node = self.gnn_node(batched_data, train)

        h_graph = self.pool(h_node, batched_data.batch)
        graph = F.relu(h_graph)
        graph=self.graph_pred_linear1(graph)
        graph = F.relu(graph)
        return self.graph_pred_linear(graph), graph, h_node




class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data,train):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr,train)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation
    



class GINConv(MessagePassing):
    def __init__(self, emb_dim = 300, aggr = "add"):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr = None, train = True):
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 7), device=edge_index.device)
        
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    


# Dirichlet energy Regularization

def compute_dirichlet_energy(x, edge_index,split):

    # adj = to_dense_adj(edge_index)[0]  # Convert edge_index to adjacency matrix
    _,count = torch.unique(edge_index[0], return_counts=True)
    # count = {key.item(): value.item() for key, value in zip(_, count)}
    di = count[edge_index[0]]
    # di =  torch.tensor([count[key.item()] for key in edge_index[0]],device = x.device)
    dj = count[edge_index[1]]
    # dj = torch.tensor([count[key.item()] for key in edge_index[1]],device = x.device)
    diff = x[edge_index[0]]/torch.sqrt(di.view(-1,1)) - x[edge_index[1]]/torch.sqrt(dj.view(-1,1))
    edge_sum = torch.sum(diff.pow(2),dim=1)
    split = split.tolist()
    split_pionts = [split[i+1] - split[i] for i in range(len(split) - 1)]
    graph_sum = torch.stack([(1/2)*torch.sum(x) for x in torch.split(edge_sum, split_pionts)]).view(-1) # x ha il gradiente graph_sum no
    return graph_sum


def DE_regularization(outputs, inputs, labels, dirichlet_bounds,  device = 'cuda'):
    '''
    Compute the Dirichlet energy regularization term.
    Take as input the output of the model, the validation loader and the model	
    '''

    dirichlet_energy = compute_dirichlet_energy(outputs, inputs.edge_index, inputs._store._slice_dict['edge_index']).to(device)
    labels = torch.argmax(labels, dim=1).to(device)
    upper_bounds = dirichlet_bounds[labels].view_as(dirichlet_energy)

    return torch.mean(torch.relu(dirichlet_energy - upper_bounds)**2)


def compute_DE_bounds(model, val_loader, class_counts, num_classes, current_bounds, batch_size = 4, device = 'cuda'):
    '''
    Compute the Dirichlet energy bounds for each class
    Returns a vector with the per class average dirichlet energy
    '''
    model.eval()
    bounds = torch.zeros(num_classes).to(device)

    for i, data in enumerate(val_loader):
        inputs, labels = data.to(device), data.y.view(-1).to(device)
        _, _, h_node = model(inputs)
        graph_sum = compute_dirichlet_energy(h_node, inputs.edge_index,inputs._store._slice_dict['edge_index']).view(-1).to(device)

        bounds.index_add_(0, labels, graph_sum)

    if len(current_bounds) == 0:
        return bounds / class_counts

    bounds = bounds / class_counts
    min_DE = torch.min(bounds, current_bounds)

    return min_DE


# Auxiliary functions 

def count_classes(data_loader, num_classes):
    '''
    Count the number of samples per class in a dataset
    '''
    class_counts = torch.zeros(num_classes)
    for data in data_loader:
        labels = data.y
        class_counts += torch.bincount(labels, minlength=num_classes)
    return class_counts


def new_alpha(class_loss, dirichlet_loss):
    '''
    Compute the new parameter alpha for balance of loss and regularization
    '''
    
    aux = class_loss / dirichlet_loss
    aux = aux * 0.1
    alpha = 10 ** round(np.log10(aux).item())
    return alpha if alpha < 0.01 else 1e-1


def set_seed(seed = 42):
    '''
    Set the seed for reproducibility
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    return seed


def save_checkpoint(model, path):
    '''
    Save the model checkpoint
    '''
    dataset = path.split('/')[1]
    torch.save(model.state_dict(), f'checkpoints/{dataset}/checkpoint.pth')

    return None