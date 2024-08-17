import torch
import torch.optim as optim
from models import PatternFlowNet
from custom_losses import MobilityLoss
from evaluation_tasks import perform_evaluation
import numpy as np

def load_data():
    mob_pattern = np.load("./Data/human_flow_p.npy")
    pattern_list = [torch.tensor(mob_pattern[i], dtype=torch.float) for i in range(mob_pattern.shape[0])]
    road = np.load('Data/path_p.npy')
    pattern_list.append(torch.tensor(road, dtype=torch.float))
    mob_adj = np.load("./Data/actual_flow.npy")
    return pattern_list, torch.Tensor(mob_adj)

def train_pattern_flow_model(input_tensor, label, criterion=None, model=None):
    if criterion is None:
        criterion = MobilityLoss()
    if model is None:
        num_branches = 8
        input_dim = 69
        hidden_dim = 128
        branch_output_dim = 120
        final_output_dim = 128
        num_heads = 8
        model = PatternFlowNet(num_branches, input_dim, hidden_dim, branch_output_dim, final_output_dim, num_heads)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    for epoch in range(1800):
        model.train()
        s_out, t_out = model(input_tensor)
        loss = criterion(s_out, t_out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch > 1750:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            embs = model.get_features()
            embs = embs.detach().numpy()
            results = perform_evaluation(embs)
            # Save results to CSV
            # file_exists = os.path.isfile('results.csv')
            # with open('results.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     if not file_exists:
            #         writer.writerow(columns)
            #     writer.writerow(results)

if __name__ == '__main__':
    pattern_list, mob_adj = load_data()
    train_pattern_flow_model(pattern_list, mob_adj)
