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

def evaluate(model_path, ptfile):
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

        f2 = fbeta_score(y_pred=y_pred, y_true=y_test, beta=2)


    print("Finish testing")

    print("f2: ", f2)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process program parameters')

    parser.add_argument('--program_language',
                        type=str,
                        default='c',
                        help='Programming language to process')
    
    parser.add_argument('--model_path',
                        type=str,
                        default='model.pt',
                        help='model pt file')
    
    parser.add_argument('--cwe',
                        type=str,
                        default='124',
                        help='cwe number')
    
    

    args = parser.parse_args()
    evaluate(args.balance_type, model_path=f"models/{args.program_language}_imbalance/{args.model_path}", ptfile=f"data/correlations/{args.program_language}_cwe-{args.cwe}.pt")