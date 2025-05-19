from train import CommitGCN
import torch
from torch_geometric.data import Data, DataLoader
from sklearn import metrics
from sklearn.metrics import fbeta_score
from utils import cost_effort_at_l, post_at_l
import numpy as np
import torch.nn.functional as F
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(balance_type, model_path, ptfile):
    model = CommitGCN(num_node_features=768, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()
    y_pred = []
    y_test = []
    probs = []


    dataset = torch.load(ptfile)
    loader = DataLoader(dataset, batch_size=32)
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
        f2 = fbeta_score(y_pred=y_pred, y_true=y_test, beta=2)
        #mcc = matthews_corrcoef(y_true=y_test, y_pred=y_pred)

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

    if balance_type == "balance":
        print(
            "precision: ", precision,
            "recall: ", recall,
            "f1: ", f1,
        )
    elif balance_type == "imbalance":
        print(
            "precision: ", precision,
            "recall: ", recall,
            "f1: ", f1,
            "f2: ", f2,
            "auc: ", auc,
            "auc_pr: ", auc_pr,
            "cost_effort_5: ", cost_effort_5,
            "cost_effort_10: ", cost_effort_10,
            "cost_effort_15: ", cost_effort_15,
            "cost_effort_20: ", cost_effort_20,
            "post_5: ", post_5,
            "post_10: ", post_10,
            "post_15: ", post_15,
            "post_20: ", post_20
        )
    return precision, recall, f1, auc, auc_pr, cost_effort_5, cost_effort_10, cost_effort_15, cost_effort_20, post_5, post_10, post_15, post_20
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process program parameters')

    parser.add_argument('--program_language',
                        type=str,
                        default='c',
                        help='Programming language to process')

    parser.add_argument('--balance_type',
                        type=str,
                        default='balance',
                        help='Balance type of the dataset')

    parser.add_argument('--data_path',
                        type=str,
                        default='data.pt',
                        help='data pt file')
    
    parser.add_argument('--model_path',
                        type=str,
                        default='model.pt',
                        help='model pt file')
    
    

    args = parser.parse_args()
    precision, recall, f1, auc, auc_pr, cost_effort_5, cost_effort_10, cost_effort_15, cost_effort_20, post_5, post_10, post_15, post_20 = evaluate(args.balance_type, model_path=f"models/{args.program_language}_{args.balance_type}/{args.model_path}", ptfile=f"data/correlations/{args.data_path}")