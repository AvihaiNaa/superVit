import torch
from torch.optim import Adam
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from VIT.VIT import MyViT
from config import CONFIG
import numpy as np

cross_entropy_loss = CrossEntropyLoss()

def train(train_loader, val_loader, api=None):
    model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(CONFIG.DEVICE)

    # Training loop
    optimizer = Adam(model.parameters(), lr=CONFIG.VIT.LR)
    criterion = CrossEntropyLoss()
    min_loss = np.inf
    for epoch in trange(CONFIG.VIT.N_EPOCHS, desc="Training"):
        train_loss = train_epoch(train_loader, device=CONFIG.DEVICE, optimizer=optimizer, model=model, api=api)
        val_loss = validation_epoch(train_loader, device=CONFIG.DEVICE, model=model, api=api)
        if api is not None:
            raise NotImplementedError
        if val_loss < min_loss:
            min_loss = val_loss
            _save_checkpoint(model, optimizer, val_loss, exp_name=CONFIG.VIT.NAME +"_"+CONFIG.VIT.DS_NAME+"_"+CONFIG.VIT.TYPE)

        print(f"Epoch {epoch + 1}/{CONFIG.VIT.N_EPOCHS} loss: {train_loss:.2f}")
    return model


def train_epoch(train_loader, device, optimizer, model, api=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        y_hat = model(data)
        if CONFIG.VIT.LOSS == "CROSS_ENTROPY": 
            loss = cross_entropy_loss(y_hat, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = loss.detach().cpu().item()
    return loss

def validation_epoch(val_loader, device, model, api=None):
    with torch.no_grad():
        model.eval()
        loss = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            y_hat = model(data)
            if CONFIG.VIT.LOSS == "CROSS_ENTROPY": 
                loss += cross_entropy_loss(y_hat, target)
        return loss

def _save_checkpoint(model, optimizer, test_loss, exp_name=''):
    """

    Args:
        model: VIT model (nn.module)
        optimizer: PyTorch optimizer class
        test_loss: float, test loss at time of saving the checkpoint
        exp_name: file name (str)
    """
    #print("saving model checkpoint")
    torch.manual_seed(1234)
    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'loss': test_loss
                  }

    torch.save(checkpoint, f'{CONFIG.TEMP_SUB_EXP_PATH}/{exp_name}_checkpoint.pth')

def _load_checkpoint(exp_name=''):
    import os
    """
    Args:
        exp_name
    """
    with torch.no_grad():
        check_path =  f'{CONFIG.TEMP_SUB_EXP_PATH}/{exp_name}_checkpoint.pth'
        if os.path.isfile(check_path):
            torch.manual_seed(1234)
            checkpoint = torch.load(check_path)
            model = checkpoint['model_state_dict']
            optimizer = checkpoint['optimizer']
            loss = checkpoint['loss']
            return model, optimizer, loss
        else:
            raise ValueError

