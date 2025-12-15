from tqdm import tqdm
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

def evaluate_model(model, train_loader, valid_loader, train_epoch, validate_epoch, optimizer, criterion, scheduler, device, models_dir, NUM_EPOCHS, PATIENCE, model_name):

    history = { 'train_loss': [], 'valid_loss': [], 'val_rmse': [], 'lr': []}

    best_val = float('inf')
    best_model = None

    epoch_counter = 0
    for epoch in range(NUM_EPOCHS): 
        print(f"\nEpoch {epoch+1} / {NUM_EPOCHS}"); print(32 * '- ')
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, rmse, pred_list, targ_list = validate_epoch(model, valid_loader, criterion, device)
        
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['val_rmse'].append(rmse)
        history['lr'].append(current_lr)
        print(f"Train_loss : {train_loss:.4f}, Valid_loss : {valid_loss:.4f}, RMSE : {rmse:.4f}, LR : {current_lr:.6f}")

        if valid_loss < best_val: 
            epoch_counter = 0
            print(f"Improvement from {best_val:.4f} to {valid_loss:.4f}")
            best_val, best_model = valid_loss, model.state_dict()
            torch.save(best_model, os.path.join(models_dir, f"best_model_{model_name}.pth")) 
        else: epoch_counter += 1; print(f"No improvement from {best_val:.4f}")
        if epoch_counter >= PATIENCE: print("Early stopping"); break
        

    return history