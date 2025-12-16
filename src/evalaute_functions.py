from tqdm import tqdm
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

@torch.no_grad()
def evaluate_testds(model, loader, device):
    model.eval()
    y_pred_list, y_targ_list = [], []

    pbar = tqdm(loader, desc='Testing')
    for img, tar in pbar:
        img = img.to(device, non_blocking=True)
        tar = tar.to(device, non_blocking=True)

        y_pred = model(img)
        
        y_pred_list.append(y_pred.cpu()) 
        y_targ_list.append(tar.cpu())

    y_pred_list = torch.cat(y_pred_list, dim=0).numpy()
    y_targ_list = torch.cat(y_targ_list, dim=0).numpy()
    return y_pred_list, y_targ_list


# def tta_evaluate(model, test_data, data_dir, transform, batch_size, device, model):

#     dataset = BiomassDS(test_data, data_dir, transform)
#     loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     y_pred_list, y_targ_list = evaluate_testds(model, loader, device)

#     return y_pred_list, y_targ_list