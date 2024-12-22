import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from tabulate import tabulate

from model import CIFAR10Net
from dataset import CIFAR10Dataset
from config import Config
from utils import train_epoch, validate

def display_model_info(model):
    print("\nModel Parameter Count:")
    summary(model, (3, 32, 32))

def main():
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Using Device: {Config.DEVICE}\n")

    train_loader, test_loader = create_dataloaders()
    model = CIFAR10Net(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    display_model_info(model)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = create_optimizer(model, train_loader)

    best_acc = 0.0
    history = []

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        test_loss, test_acc = validate(model, test_loader, criterion)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

        print_epoch_summary(epoch, train_loss, train_acc, test_loss, test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    display_training_summary(history, best_acc)

def create_dataloaders():
    train_dataset = CIFAR10Dataset(root=Config.DATA_ROOT, train=True)
    test_dataset = CIFAR10Dataset(root=Config.DATA_ROOT, train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader

def create_optimizer(model, train_loader):
    params = [
        {'params': model.conv1.parameters(), 'lr': Config.LEARNING_RATE * 2.0},
        {'params': model.conv2.parameters(), 'lr': Config.LEARNING_RATE * 1.5},
        {'params': model.conv3.parameters(), 'lr': Config.LEARNING_RATE * 1.0},
        {'params': model.conv4.parameters(), 'lr': Config.LEARNING_RATE * 0.8},
        {'params': model.fc.parameters(), 'lr': Config.LEARNING_RATE * 1.5},
    ]
    optimizer = optim.Adam(params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = None
    if Config.ONE_CYCLE_LR:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.MAX_LR,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(train_loader),
            div_factor=Config.DIV_FACTOR,
            pct_start=Config.PCT_START
        )
    return optimizer, scheduler

def print_epoch_summary(epoch, train_loss, train_acc, test_loss, test_acc):
    print(f"Epoch: {epoch + 1}/{Config.EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print("-" * 50)

def display_training_summary(history, best_acc):
    print("\nTraining Summary:")
    headers = ['Epoch', 'Train Loss', 'Train Acc(%)', 'Test Loss', 'Test Acc(%)']
    table = [[
        h['epoch'],
        f"{h['train_loss']:.4f}",
        f"{h['train_acc']:.2f}",
        f"{h['test_loss']:.4f}",
        f"{h['test_acc']:.2f}"
    ] for h in history]
    print(tabulate(table, headers=headers, tablefmt='grid'))
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
