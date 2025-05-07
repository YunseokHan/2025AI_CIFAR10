import os

from datasets import CIFAR10Variant
from models import get_model

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


def trainer(cfg, use_wandb):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if use_wandb:
        import wandb
        wandb.login(key=use_wandb)
        wandb.init(
            project="CIFAR10 image classification",
            group=f"{cfg['model']} (pretrained)",
            name=cfg['tag'],
            settings={
                "_service_wait": 600,
                "init_timeout": 600
            }
        )

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    train_ds = CIFAR10Variant('./data', True, tf, **cfg['dataset'])
    test_ds  = CIFAR10Variant('./data', False, tf)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds , batch_size=256, shuffle=False, num_workers=4)

    model = get_model(cfg['model']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    os.makedirs(f"./results/{cfg['model']}/{cfg['tag']}", exist_ok=True)
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss_total = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            train_loss_total += loss.item() * y.size(0)
            optimizer.step()
        train_loss_avg = train_loss_total / len(train_loader.dataset)
        
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x.to(device))
                val_loss_total += criterion(outputs, y.to(device)).item() * y.size(0)
        val_loss = val_loss_total / len(test_loader.dataset)
        print(f"[{epoch+1}/{cfg['epochs']}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss:.4f}")
        if use_wandb:
            wandb.log({'train_loss': train_loss_avg, 'val_loss': val_loss, 'epoch': epoch + 1})
        else:
            with open(f"./results/{cfg['model']}/{cfg['tag']}/log.txt", 'a') as f:
                f.write(f"Epoch {epoch+1}, Train Loss: {train_loss_avg}, Val Loss: {val_loss}\n")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"./results/{cfg['model']}/{cfg['tag']}/best_model.pt")
            print(f"New best model saved with val_loss: {best_loss:.4f}")
    if use_wandb:
        wandb.finish()

__all__ = ["train"]