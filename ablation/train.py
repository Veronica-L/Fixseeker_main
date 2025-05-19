import os.path
from sklearn import metrics
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import networkx as nx
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
from itertools import islice
import logging
from imblearn.over_sampling import RandomOverSampler
from torch.nn.functional import cross_entropy
from sklearn.metrics import matthews_corrcoef
from utils import cost_effort_at_l, post_at_l
import argparse
import re
import Levenshtein as lev
from utils import normalize_code, abstract_code
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Process program parameters')

    parser.add_argument('--program_language',
                        type=str,
                        default='c',
                        help='Programming language to process')

    parser.add_argument('--balance_type',
                        type=str,
                        default='imbalance',
                        help='Balance type of the dataset')
    
    parser.add_argument('--ablation_type',
                        type=str,
                        default='no_edges',
                        choices=['no_edges', 'no_call', 'no_ddg', 'no_cfg', 'no_sim'],
                        help='Ablation type: remove all edges or specific edge type')

    args = parser.parse_args()
    return args

args = parse_args()
LANGUAGE = args.program_language
IF_BALANCE = args.balance_type
ABLATION_TYPE = args.ablation_type
DATASET = f'models/{LANGUAGE}_{IF_BALANCE}.json'
PTFILE = f"data/correlations/{LANGUAGE}_{IF_BALANCE}.pt"
MODEL_SAVE_PATH = f"models/{LANGUAGE}_{IF_BALANCE}_ablation_{ABLATION_TYPE}"

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# 修改日志文件名
logging.basicConfig(filename=f'log/{LANGUAGE}_{IF_BALANCE}_ablation_{ABLATION_TYPE}.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = RobertaTokenizer.from_pretrained("./microsoft/codebert-base")
codebert_model = RobertaModel.from_pretrained("./microsoft/codebert-base")

def calculate_class_weights(dataset):
    labels = [data.y.item() for data in dataset]
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    return class_weights

def oversample_graph_dataset(dataset):
    labels = [data.y.item() for data in dataset]

    X = np.array([[i] for i in range(len(dataset))])

    oversampler = SMOTE(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, labels)

    oversampled_dataset = []
    for idx in X_resampled:
        oversampled_dataset.append(dataset[idx[0]])

    return oversampled_dataset


# 边消融相关函数
def remove_all_edges(dataset):
    print("[Remove all edges]")
    processed_dataset = []
    
    for data in tqdm(dataset, desc="Processing datasets"):
        empty_edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        
        new_data = Data(
            x=data.x,
            edge_index=empty_edge_index,
            y=data.y
        )
        processed_dataset.append(new_data)
    
    print(f"Finish processing, data length: {len(processed_dataset)}")
    return processed_dataset

def remove_edge_type(dataset, edge_type_to_remove):
    edge_type_map = {
        'no_call': 0,  # CALL
        'no_ddg': 1,   # DDG
        'no_cfg': 2,   # CFG
        'no_sim': 3    # SIM
    }
    
    target_type = edge_type_map[edge_type_to_remove]
    print(f"[Remove edges of type: {edge_type_to_remove} (index {target_type})]")
    
    processed_dataset = []
    
    for data in tqdm(dataset, desc="Processing datasets"):
        if hasattr(data, 'edge_type') and data.edge_index.size(1) > 0:
            mask = data.edge_type != target_type
            
            new_edge_index = data.edge_index[:, mask]
            new_edge_type = data.edge_type[mask]
            
            new_data = Data(
                x=data.x,
                edge_index=new_edge_index,
                edge_type=new_edge_type,
                y=data.y
            )
        else:
            new_data = data
            
        processed_dataset.append(new_data)
    
    print(f"Finish processing, data length: {len(processed_dataset)}")
    return processed_dataset

def perform_ablation(dataset, ablation_type):
    if ablation_type == 'no_edges':
        return remove_all_edges(dataset)
    elif ablation_type in ['no_call', 'no_ddg', 'no_cfg', 'no_sim']:
        return remove_edge_type(dataset, ablation_type)
    else:
        print(f"Unknown ablation type: {ablation_type}, returning original dataset")
        return dataset


class CommitGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(CommitGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, num_classes)
        self.no_edge_transform = torch.nn.Linear(num_node_features, 64)

    def forward(self, x, edge_index, batch):
        if edge_index.numel() == 0:
            x = F.relu(self.no_edge_transform(x))
            x = global_mean_pool(x, batch)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)


def train(model, epoch, trainloader, valloader, testloader, optimizer, device, class_weights):
    model.train()
    train_loss, val_loss = 0, 0
    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = cross_entropy(out, data.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.num_graphs

    print("epoch {}, training commit loss {}".format(epoch, np.sum(train_loss / len(trainloader.dataset))))

    model.eval()
    for data in valloader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = cross_entropy(out, data.y, weight=class_weights)
        val_loss += loss.item() * data.num_graphs

    print("epoch {}, evaluate commit loss {}".format(epoch, np.sum(val_loss / len(valloader.dataset))))

    precision, recall, f1, auc, auc_pr, cost_effort_5, cost_effort_10, cost_effort_15, cost_effort_20, post_5, post_10, post_15, post_20 = evaluate(model, testloader, device, class_weights)

    print(f'PR: {precision}, RE: {recall}, F1: {f1}')
    print("AUC-ROC: {}".format(auc))
    print("AUC-PR: {}".format(auc_pr))
    print("COST-EFFORT@5: {}".format(cost_effort_5))
    print("COST-EFFORT@10: {}".format(cost_effort_10))
    print("COST-EFFORT@15: {}".format(cost_effort_15))
    print("COST-EFFORT@20: {}".format(cost_effort_20))
    print("POST@5: {}".format(post_5))
    print("POST@10: {}".format(post_10))
    print("POST@15: {}".format(post_15))
    print("POST@20: {}".format(post_20))

    return precision, recall, f1, auc, auc_pr, cost_effort_5, cost_effort_10, cost_effort_15, cost_effort_20, post_5, post_10, post_15, post_20

# evaluate function
def evaluate(model, loader, device, class_weights):
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    y_test = []
    probs = []

    with torch.no_grad():
        model.eval()
        for data in loader:
            data = data.to(device)
            outs = model(data.x, data.edge_index, data.batch)

            outs = F.softmax(outs, dim=1)

            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(data.y.tolist())
            probs.extend(outs[:, 1].tolist())

        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

        cost_effort_5 = cost_effort_at_l(np.array(y_test), np.array(probs), l=0.05)
        post_5 = post_at_l(np.array(y_test), np.array(probs), l=0.05)
        cost_effort_10 = cost_effort_at_l(np.array(y_test), np.array(probs), l=0.1)
        post_10 = post_at_l(np.array(y_test), np.array(probs), l=0.1)
        cost_effort_15 = cost_effort_at_l(np.array(y_test), np.array(probs), l=0.15)
        post_15 = post_at_l(np.array(y_test), np.array(probs), l=0.15)
        cost_effort_20 = cost_effort_at_l(np.array(y_test), np.array(probs), l=0.2)
        post_20 = post_at_l(np.array(y_test), np.array(probs), l=0.2)

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
            auc_pr = metrics.average_precision_score(y_score=probs, y_true=y_test)
        except Exception:
            auc = 0
            auc_pr = 0

    print("Finish testing")
    return precision, recall, f1, auc, auc_pr, cost_effort_5, cost_effort_10, cost_effort_15, cost_effort_20, post_5, post_10, post_15, post_20


def main():
    ptfile = PTFILE
    ablation_ptfile = PTFILE.replace('.pt', f'_ablation_{ABLATION_TYPE}.pt')  # 创建一个新的PT文件以区分消融实验数据
    
    if os.path.exists(ablation_ptfile):
        print(f"Loading data: {ablation_ptfile}")
        dataset = torch.load(ablation_ptfile)
    else:
        print(f"Creating ablation data for {ABLATION_TYPE}...")
        if os.path.exists(ptfile):
            print(f"Loading data from original file: {ptfile}")
            original_dataset = torch.load(ptfile)
            
            dataset = perform_ablation(original_dataset, ABLATION_TYPE)
            torch.save(dataset, ablation_ptfile)
        else:
            print(f"Error: file {ptfile} does not exist!")
            return

    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=40
    )
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.2 / (1 - 0.2), random_state=40
    )
    
    if not os.path.exists(f'model_pt/mydata_pt/{LANGUAGE}_{IF_BALANCE}_ablation_{ABLATION_TYPE}_test.pt'):
        torch.save(test_dataset, f'model_pt/mydata_pt/{LANGUAGE}_{IF_BALANCE}_ablation_{ABLATION_TYPE}_test.pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if IF_BALANCE == 'imbalance':
        class_weights = calculate_class_weights(train_dataset)
        class_weights = class_weights.to(device)

        train_dataset = oversample_graph_dataset(train_dataset)
        val_dataset = oversample_graph_dataset(val_dataset)
    else:
        class_weights = None

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = CommitGCN(num_node_features=768, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("[Start training]")
    best_auc, best_auc_pr, best_f1, best_mcc = 0, 0, 0, -1
    best_costeffort_5, best_costeffort_10, best_costeffort_15, best_costeffort_20 = 0,0,0,0
    best_post_5, best_post_10, best_post_15, best_post_20 = 0,0,0,0
    for epoch in range(500):
        print(f"\n=========== Ablation ({ABLATION_TYPE}) - epoch {epoch} ===========")
        precision, recall, f1, auc, auc_pr, cost_effort_5, cost_effort_10, cost_effort_15, cost_effort_20, post_5, post_10, post_15, post_20 = train(model, epoch, train_loader, val_loader, test_loader, optimizer, device, class_weights=class_weights)

        if IF_BALANCE == 'imbalance':
            if auc > best_auc:
                best_auc = auc
                best_model = model.state_dict()
                torch.save(best_model, f'{MODEL_SAVE_PATH}/best_model_epoch_{epoch}.pt')
                print(f'New best model saved with AUC: {best_auc:.4f}')
                logging.info(f'New best model saved with AUC: {best_auc:.4f} in epoch {epoch}')
            if auc_pr > best_auc_pr:
                best_auc_pr = auc_pr
                best_model = model.state_dict()
                torch.save(best_model, f'{MODEL_SAVE_PATH}/best_model_epoch_{epoch}.pt')
                print(f'New best model saved with AUC-PR: {best_auc_pr:.4f}')
                logging.info(f'New best model saved with AUC-PR: {best_auc_pr:.4f} in epoch {epoch}')
            if cost_effort_5 > best_costeffort_5:
                best_costeffort_5 = cost_effort_5
                print(f'New best model saved with CostEffot@5: {best_costeffort_5:.4f}')
                logging.info(f'New best model saved with CostEffort@5: {best_costeffort_5:.4f} in epoch {epoch}')
            if cost_effort_10 > best_costeffort_10:
                best_costeffort_10 = cost_effort_10
                print(f'New best model saved with CostEffot@10: {best_costeffort_10:.4f}')
                logging.info(f'New best model saved with CostEffort@10: {best_costeffort_10:.4f} in epoch {epoch}')
            if cost_effort_15 > best_costeffort_15:
                best_costeffort_15 = cost_effort_15
                print(f'New best model saved with CostEffot@15: {best_costeffort_15:.4f}')
                logging.info(f'New best model saved with CostEffort@15: {best_costeffort_15:.4f} in epoch {epoch}')
            if cost_effort_20 > best_costeffort_20:
                best_costeffort_20 = cost_effort_20
                print(f'New best model saved with CostEffot@20: {best_costeffort_20:.4f}')
                logging.info(f'New best model saved with CostEffort@20: {best_costeffort_20:.4f} in epoch {epoch}')

            if post_5 > best_post_5:
                best_post_5 = post_5
                print(f'New best model saved with Post@5: {best_post_5:.4f}')
                logging.info(f'New best model saved with Post@5: {best_post_5:.4f} in epoch {epoch}')
            if post_10 > best_post_10:
                best_post_10 = post_10
                print(f'New best model saved with Post@10: {best_post_10:.4f}')
                logging.info(f'New best model saved with Post@10: {best_post_10:.4f} in epoch {epoch}')
            if post_15 > best_post_15:
                best_post_15 = post_15
                print(f'New best model saved with Post@15: {best_post_15:.4f}')
                logging.info(f'New best model saved with Post@15: {best_post_15:.4f} in epoch {epoch}')
            if post_20 > best_post_20:
                best_post_20 = post_20
                print(f'New best model saved with Post@20: {best_post_20:.4f}')
                logging.info(f'New best model saved with Post@20: {best_post_20:.4f} in epoch {epoch}')
        else:
            if f1 > best_f1:
                best_f1 = f1
                best_model = model.state_dict()
                torch.save(best_model, f'{MODEL_SAVE_PATH}/best_model_epoch_{epoch}.pt')
                print(f'New best model saved with F1: {best_f1:.4f}, Precision: {precision:.4f}, Recall:{recall:.4f}')
                logging.info(f'New best model saved with F1: {best_f1:.4f} in epoch {epoch}')


if __name__ == "__main__":
    main()