import argparse
from comet_ml import Experiment
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from loadData import GraphDataset
import os
import pandas as pd
from tqdm import tqdm

from first_submission import GNN, DE_regularization, compute_DE_bounds, count_classes, new_alpha, set_seed, save_checkpoint


experiment = Experiment(
    api_key="saeTJTVrQrbdwDxDGTv8lLRIL",
    project_name="noisy-graph-labels-challenge",
    workspace="mavivestini"
)



def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data


class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.embedding = torch.nn.Embedding(1, input_dim) 
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.global_pool = global_mean_pool  
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)  
        out = self.fc(x)  
        return out


def train(data_loader, dirichlet_bounds, epoch, alpha):
    model.train()
    total_loss = 0
    dirichlet_reg = 0
    total_dirichlet_reg = 0
    class_loss = 0
    steps_to_accumulate = 8
    for i, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        outputs, _, h_node = model(data)

        if epoch > 10:
            dirichlet_reg = DE_regularization(h_node, data, outputs, dirichlet_bounds)
            total_dirichlet_reg += dirichlet_reg.item()
        

        loss = criterion(outputs, data.y)
        class_loss += loss.item()
        loss += alpha * dirichlet_reg
        loss = loss / steps_to_accumulate
        loss.backward()

        if i % steps_to_accumulate == 0 or i == len(data_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(data_loader), class_loss / len(data_loader), total_dirichlet_reg / len(data_loader)


def evaluate(data_loader, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs, _, _ = model(data)
            pred = outputs.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions


def main(args):
    global model, optimizer, criterion, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed()

    # Parameters for the GCN model
    input_dim = 300  # Example input feature dimension (you can adjust this)
    hidden_dim = 64
    output_dim = 6  # Number of classes

    # Initialize the model, optimizer, and loss criterion
    model = GNN(num_class = output_dim, num_layer = 5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train dataset and loader (if train_path is provided)
    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)

        # Split the training dataset into training and validation sets
        num_train = int(0.8 * len(train_dataset))
        num_val = len(train_dataset) - num_train
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Training loop
        num_epochs = 200
        alpha = 1e-9
        best_acc = 0
        dirichlet_bounds = []
        class_counts = count_classes(val_loader, output_dim).to(device)
        for epoch in range(num_epochs):
            if epoch >= 10:
                with torch.no_grad():
                    dirichlet_bounds = compute_DE_bounds(model, val_loader, class_counts, output_dim, dirichlet_bounds)
            train_loss, class_loss, dirichlet_loss = train(train_loader, dirichlet_bounds, epoch, alpha)
            if epoch > 10 and dirichlet_loss != 0 and epoch % 20 == 0:
                alpha = new_alpha(class_loss, dirichlet_loss)
            train_acc, _ = evaluate(val_loader, calculate_accuracy=True)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("class_loss", class_loss, step=epoch)
            experiment.log_metric("dirichlet_loss", dirichlet_loss, step=epoch)
            experiment.log_metric("val_acc", train_acc, step=epoch)

            if best_acc < train_acc:
                save_checkpoint(model, args.train_path)
                best_acc = train_acc

    else:
        dataset = args.test_path.split('/')[1]
        state_dict_ = torch.load(f'checkpoints/{dataset}/checkpoint.pth')
        model.load_state_dict(state_dict_)

    # Evaluate and save test predictions
    predictions = evaluate(test_loader, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "GraphID": test_graph_ids,
        "Class": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    args = parser.parse_args()
    main(args)


# python main.py --train_path data/A/train.json.gz --test_path data/A/test.json.gz  --batch_size 4
# python main.py --train_path data/B/train.json.gz --test_path data/B/test.json.gz  --batch_size 4
# python main.py --train_path data/C/train.json.gz --test_path data/C/test.json.gz  --batch_size 4
# python main.py --train_path data/D/train.json.gz --test_path data/D/test.json.gz  --batch_size 4