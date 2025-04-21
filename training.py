import torch
import torch.optim as optim
from models import MSIN
from custom_losses import MobilityLoss
from evaluation_tasks import perform_evaluation
import numpy as np

def load_Data_180():
    mob_pattern = np.load("./Data_180/human_flow_p.npy")
    pattern_list = [torch.tensor(mob_pattern[i], dtype=torch.float) for i in range(mob_pattern.shape[0])]
    road = np.load('Data_180/path_p.npy')
    pattern_list.append(torch.tensor(road, dtype=torch.float))
    mob_adj = np.load("./Data_180/actual_flow.npy")
    return pattern_list, torch.Tensor(mob_adj)

def train(input_tensor, label, criterion=None, model=None):
    b_check_r2 = 0
    b_crime_r2 = 0
    b_call_r2 = 0
    if criterion is None:
        criterion = MobilityLoss()
    if model is None:
        num_branches = 8
        #region 
        input_dim = 180
        hidden_dim = 128
        branch_output_dim = 120
        final_output_dim = 128
        num_heads = 8
        epochs = 2000
        model = MSIN(num_branches, input_dim, hidden_dim, branch_output_dim, final_output_dim, num_heads)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        s_out, t_out = model(input_tensor)
        loss = criterion(s_out, t_out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        embs = model.get_features()
        embs = embs.detach().numpy()

        if epoch %25 == 0:
            print(f"\nEpoch {epoch}, Loss {loss.item()}")
            pop_mae, pop_rmse, pop_r2,cri_mae, cri_rmse, cri_r2, call_mae, call_rmse, call_r2,check_mae, check_rmse, check_r2, nmi, ars = perform_evaluation(embs,True)
        else:
            pop_mae, pop_rmse, pop_r2,cri_mae, cri_rmse, cri_r2, call_mae, call_rmse, call_r2,check_mae, check_rmse, check_r2, nmi, ars = perform_evaluation(embs,False)
            # Save results to CSV
            # file_exists = os.path.isfile('results.csv')
            # with open('results.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     if not file_exists:
            #         writer.writerow(columns)
            #     writer.writerow(results)
        if cri_r2>b_crime_r2:
            b_crime_r2 = cri_r2
        if call_r2>b_call_r2:
            b_call_r2=call_r2
        if check_r2>b_check_r2:
            b_check_r2=check_r2
    print("### Best ###")
    print(f"check-in Prediction -  R2: {b_check_r2:.4f}")
    print(f"crime Prediction - R2: {b_crime_r2:.4f}")
    print(f"call Prediction - R2: {b_call_r2:.4f}")


if __name__ == '__main__':
    pattern_list, mob_adj = load_Data_180()
    train(pattern_list, mob_adj)
