from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error



def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss, num_batches = 0, 0

    pbar = tqdm(loader, desc='Training')
    for img, tar in pbar:
        img = img.to(device); tar = tar.to(device)

        optimizer.zero_grad()

        pred = model(img)
        loss = criterion(pred, tar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item(); num_batches += 1
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches; 
    return avg_loss

def validate_epoch(model, loader, criterion, device):
    model.eval()

    total_loss, num_batches = 0, 0

    all_preds, all_targs = [], []

    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for img, tar in pbar:
            img = img.to(device); tar = tar.to(device)

            pred = model(img)
            loss = criterion(pred, tar); 
            
            total_loss += loss.item(); num_batches += 1

            all_preds.append(pred.cpu()) 
            all_targs.append(tar.cpu())
            
            pbar.set_postfix(loss=loss.item())

    all_preds   = torch.cat(all_preds).numpy() 
    all_targs   = torch.cat(all_targs).numpy()

    avg_loss    = total_loss / num_batches 
    rmse        = np.sqrt(mean_squared_error(all_targs, all_preds))
    
    return avg_loss, rmse, all_preds, all_targs

def train_epoch_twostream(model, loader, optimizer, criterion, device):

    model.train()

    total_loss, num_batches = 0, 0

    pbar = tqdm(loader, desc='Training')
    for left, right, target in pbar:
        left, right, target = left.to(device), right.to(device), target.to(device)

        optimizer.zero_grad()

        pred = model(left, right)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item(); num_batches += 1
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches; 
    return avg_loss

def validate_epoch_twostream(model, loader, criterion, device):
    model.eval()

    total_loss, num_batches = 0, 0

    all_preds, all_targs = [], []

    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for left, right, target in pbar:
            left, right, target = left.to(device), right.to(device), target.to(device)

            pred = model(left, right)
            loss = criterion(pred, target); 
            
            total_loss += loss.item(); num_batches += 1

            all_preds.append(pred.cpu()) 
            all_targs.append(target.cpu())
            
            pbar.set_postfix(loss=loss.item())

    all_preds   = torch.cat(all_preds).numpy() 
    all_targs   = torch.cat(all_targs).numpy()

    avg_loss    = total_loss / num_batches 
    rmse        = np.sqrt(mean_squared_error(all_targs, all_preds))
    
    return avg_loss, rmse, all_preds, all_targs


def train_epoch_threehead(model, loader, optimizer, criterion, device):
    model.train()
    train_loss, num_batches = 0, 0

    pbar = tqdm(loader, desc='Training')
    for img, targets in pbar:
        img = img.to(device); targets = targets.to(device)
        tar_green, tar_total, tar_gdm = targets[:,0], targets[:,1], targets[:,2]
        pred_green, pred_total, pred_gdm = model(img)
        
        # Dry_Green_g, Dry_Total_g, GDM_g


        optimizer.zero_grad()

        green_loss = criterion(pred_green.squeeze(), tar_green)
        total_loss = criterion(pred_total.squeeze(), tar_total)
        gdm_loss   = criterion(pred_gdm.squeeze(), tar_gdm); 
        
        loss = green_loss + total_loss + gdm_loss; 

        loss.backward()
        optimizer.step()

        train_loss += loss.item(); num_batches += 1; 
        pbar.set_postfix(loss=loss.item())
    return train_loss / num_batches

def validate_epoch_threehead(model, loader, criterion, device):
    model.eval()
    valid_loss, num_batches = 0, 0
    pred_list , targ_list = [], []

    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for img, targets in pbar:
            img = img.to(device); targets = targets.to(device)
            tar_green, tar_total, tar_gdm = targets[:,0], targets[:,1], targets[:,2]
            # Dry_Green_g, Dry_Total_g, GDM_g

            pred_green, pred_total, pred_gdm = model(img)
            
            green_loss = criterion(pred_green.squeeze(), tar_green)
            total_loss = criterion(pred_total.squeeze(), tar_total)
            gdm_loss   = criterion(pred_gdm.squeeze(), tar_gdm)
            loss = green_loss + total_loss + gdm_loss
            valid_loss += loss.item(); num_batches += 1

            preds = torch.stack([pred_green.squeeze(), pred_total.squeeze(), pred_gdm.squeeze()], dim=1)
            pred_list.append(preds.cpu())
            targ_list.append(targets.cpu())
            
            pbar.set_postfix(loss=loss.item())

    avg_loss = valid_loss / num_batches
    pred_list = torch.cat(pred_list).numpy()
    targ_list = torch.cat(targ_list).numpy()     
    rmse = np.sqrt(mean_squared_error(targ_list, pred_list))

    return avg_loss, rmse, pred_list, targ_list