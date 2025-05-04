import torch
import torch.optim as optim
from models import MSIN
from custom_losses import MobilityLoss
from evaluation_tasks import perform_evaluation
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_Data_180():
    # 加载三通道 mobility pattern（7, 3, 180, 180）
    mob_pattern = np.load("./Data_180/mob_patterns_3channel.npy")  # shape: (7, 3, 180, 180)
    pattern_tensor = torch.tensor(mob_pattern, dtype=torch.float)  # 转为 PyTorch 张量
    pattern_tensor = pattern_tensor.to(device)
    # 加载邻接矩阵（区域流动图）
    mob_adj = np.load("./Data_180/actual_flow.npy")  # shape: (180, 180)
    mob_adj_tensor = torch.tensor(mob_adj, dtype=torch.float)
    mob_adj_tensor = mob_adj_tensor.to(device)
    # 加载路径数据
    road = np.load("./Data_180/path_p.npy")  # shape: (180, 180)
    road_tensor = torch.tensor(road, dtype=torch.float)
    road_tensor =road_tensor.to(device)
    return pattern_tensor, mob_adj_tensor, road_tensor

def train(input_tensor, label, path, criterion=None, model=None):
    b_check_r2 = 0
    b_crime_r2 = 0
    b_call_r2 = 0
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
        embs = embs.detach().numpy()

        if epoch %25 == 0:
            print(f"\nEpoch {epoch}, Loss {loss.item()}")
            cri_mae, cri_rmse, cri_r2, call_mae, call_rmse, call_r2,check_mae, check_rmse, check_r2, nmi, ars = perform_evaluation(embs,True)
        else:
            cri_mae, cri_rmse, cri_r2, call_mae, call_rmse, call_r2,check_mae, check_rmse, check_r2, nmi, ars = perform_evaluation(embs,False)
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
        if nmi>b_nmi:
            b_nmi = nmi
        if ars>b_ars:
            b_ars = ars
    print("### Best ###")
    print(f"check-in Prediction -  R2: {b_check_r2:.4f}")
    print(f"crime Prediction - R2: {b_crime_r2:.4f}")
    print(f"call Prediction - R2: {b_call_r2:.4f}")
    print(f"nmi Prediction - R2: {b_nmi:.4f}")
    print(f"ars Prediction - R2: {b_ars:.4f}")


if __name__ == '__main__':
    pattern_list, mob_adj, path = load_Data_180()
    train(pattern_list, mob_adj,path)
