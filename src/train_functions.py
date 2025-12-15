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