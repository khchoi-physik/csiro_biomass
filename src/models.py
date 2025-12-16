import torch
import torch.nn as nn

class get_dino(nn.Module):

    def __init__(self, backbone, patch_dim, num_targets=3):
        super().__init__()
        self.backbone = backbone  
        self.encoder_dim = 256
        for params in self.backbone.parameters(): params.requires_grad = False
        
        self.heads = nn.Sequential(
            nn.Linear(patch_dim, 512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.3),
            nn.Linear(512, 256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(0.3),
            nn.Linear(256, num_targets))

    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x)
            patch_features = outputs.last_hidden_state[:, 1:, :] ### (batch_size, 256, 384)
        b, n, d = patch_features.shape 
        patch_preds = self.heads(patch_features.reshape(-1, d)).reshape(b, n, -1)  
        avg_pred = torch.mean(patch_preds, dim=1) ### (batch_size, num_targets)
        return avg_pred


class get_three_head_dino(nn.Module):
    def __init__(self, backbone, patch_dim, num_targets=3):
        super().__init__()
        self.backbone = backbone  
        for params in self.backbone.parameters(): params.requires_grad = False
    
        self.encoder_dim = 256
    
        self.encoder = nn.Sequential(
            nn.Linear(patch_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3) )

        self.green_head = self.create_head(self.encoder_dim, dropout_rate=0.3)
        self.total_head = self.create_head(self.encoder_dim, dropout_rate=0.3) 
        self.gdm_head   = self.create_head(self.encoder_dim, dropout_rate=0.3) 
       
    def create_head(self, input_features, dropout_rate):
        """Create single-output prediction head"""
        return nn.Sequential(
            nn.Linear(input_features, input_features // 2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(input_features // 2, 1))

    def forward(self, x):
 
        with torch.no_grad():
            outputs = self.backbone(x).last_hidden_state[:, 1:, :] ### (batch_size, 256, 384)
        b, n, d = outputs.shape 
        encoded_features = self.encoder(outputs.reshape(-1, d)).reshape(b, n, -1) 
        pooled = torch.mean(encoded_features, dim=1) ### (batch_size, encoder_dim)

        pred_green = self.green_head(pooled)
        pred_total = self.total_head(pooled) 
        pred_gdm   = self.gdm_head(pooled)

        return pred_green, pred_total, pred_gdm
