import torch
import torch.optim as optim
from models import MSIN
from custom_losses import MobilityLoss
from evaluation_tasks import perform_evaluation
from parse_args import args
import numpy as np
from data_utils import load_Data_180

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def train(input_tensor, label, path, criterion=None, model=None):
    b_check_r2,b_check_mae,b_check_rmse = 0,0,0
    b_crime_r2,b_cri_mae,b_cri_mae = 0,0,0
    b_call_r2,b_call_mae,b_call_rmse = 0,0,0
    b_nmi = 0
    b_ars = 0
    if criterion is None:
        criterion = MobilityLoss().to(device)
    if model is None:
        num_branches = 8
        #region 
        input_dim = 180
        hidden_dim = 144
        branch_output_dim = 144
        final_output_dim = 144
        num_heads = 8
        epochs = 2000
        model = MSIN(num_branches, input_dim, hidden_dim, branch_output_dim, final_output_dim, num_heads).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    for epoch in range(epochs):
        model.train()
        s_out, t_out = model(input_tensor,path)
        loss = criterion(s_out, t_out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        embs = model.get_features()
        embs = embs.detach().cpu().numpy()

        if epoch %50 == 0:
            print(f"\nEpoch {epoch}, Loss {loss.item()}")
            cri_mae, cri_rmse, cri_r2, call_mae, call_rmse, call_r2,check_mae, check_rmse, check_r2 = perform_evaluation(embs,True)
        else:
            cri_mae, cri_rmse, cri_r2, call_mae, call_rmse, call_r2,check_mae, check_rmse, check_r2 = perform_evaluation(embs,False)
            # Save results to CSV
            # file_exists = os.path.isfile('results.csv')
            # with open('results.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     if not file_exists:
            #         writer.writerow(columns)
            #     writer.writerow(results)
        if cri_r2>b_crime_r2:
            b_crime_r2 = cri_r2
            b_cri_mae = cri_mae
            b_cri_rmse = cri_rmse
        if call_r2>b_call_r2:
            b_call_r2=call_r2
            b_call_mae = call_mae
            b_call_rmse = call_rmse
        if check_r2>b_check_r2:
            b_check_r2=check_r2
            b_check_rmse = check_rmse
            b_check_mae = check_mae

    print("### Best ###")
    print(f"check-in Prediction - MAE:{b_check_mae:.4f} , RMSE:{b_check_rmse:.4f}  R2: {b_check_r2:.4f}")
    print(f"crime Prediction - MAE:{b_cri_mae:.4f} , RMSE:{b_cri_rmse:.4f} R2: {b_crime_r2:.4f}")
    print(f"call Prediction - MAE:{b_call_mae:.4f} , RMSE:{b_call_rmse:.4f} R2: {b_call_r2:.4f}")


if __name__ == '__main__':
    pattern_list, mob_adj, path = load_Data_180()
    train(pattern_list, mob_adj,path)
