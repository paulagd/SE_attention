from torchvision.datasets import CIFAR10
from model import LeNet
import torch
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import SGD
import argparse
from datetime import datetime
import os
from IPython import embed


def parse_args():
    # srun -p veu -c8 --mem 30G python main.py
    parser = argparse.ArgumentParser()
    # parser.add_argument("--onehot", action="store_true", help="enables one-hot encoding")
    # parser.add_argument("--cuda", action="store_true", help="enables cuda")
    # parser.add_argument("--balance", action="store_true", help="enables balance data")
    # parser.add_argument("--onlytrain", action="store_true", help="enables balance data")
    # parser.add_argument("--num_workers", type=int, default=4, help="Num cpu workers")
    # parser.add_argument("--seed", type=int, default=5, help="Random seed")
    # parser.add_argument("--hidden_size", type=int, default=128, help="Random seed")
    # parser.add_argument("--layers", type=int, default=2, help="Random seed")
    # parser.add_argument("--dropout", type=float, default=0, help="dropout parameter")
    parser.add_argument("--nologs", action="store_true", help="enables no tensorboard")

    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':

    root = './data'
    # reduction_factor = 16
    # se_model = SqueezeExcitationModule(base_model, , reduction_factor)
    batch_size = 1024
    epochs = 100
    opt = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    date = datetime.now().strftime('%y%m%d%H%M%S')
    if opt.nologs:
        writer = SummaryWriter(log_dir=f'logs/nologs/')
    else:
        writer = SummaryWriter(log_dir=f'logs/balanced_logs_{date}/')

    model = LeNet().to(device)  # N x 3 x 32 x 32

    trans = transforms.Compose([transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_set = CIFAR10(root=root, train=True, transform=trans, download=True)
    test_set = CIFAR10(root=root, train=False, transform=trans, download=True)

    train_loader = DataLoader(
        dataset=train_set,
        pin_memory=device == 'cuda',
        batch_size=batch_size,
        shuffle=True)

    test_loader = DataLoader(
        dataset=test_set,
        pin_memory=device == 'cuda',
        batch_size=batch_size,
        shuffle=False)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            # x, target = Variable(x), Variable(target)
            out = model(x)
            loss = criterion(out, target)
            # embed()
            _, pred_label = torch.max(out, 1)
            total_cnt = x.size(0)
            correct_cnt = (pred_label == target).sum()
            acc_train = correct_cnt.float() / total_cnt

            ave_loss = ave_loss * 0.9 + loss.item() * 0.1
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', ave_loss, epoch*len(train_loader)+batch_idx)
            writer.add_scalar('train/acc', acc_train, epoch*len(train_loader)+batch_idx)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, batch_idx + 1, ave_loss, acc_train))

        checkpoint = {
            "state_dict": model.module.state_dict() if device == 'cuda' else model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        os.makedirs("./weights/{}".format(date), exist_ok=True)
        torch.save(checkpoint, "./weights/{}/checkpoint_{}.pt".format(date, epoch))

        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        acc_val = []
        for batch_idx, (x, target) in enumerate(test_loader):
            x = x.to(device)
            target = target.to(device)
            # x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out, 1)
            total_cnt = x.size(0)
            correct_cnt = (pred_label == target).sum()
            acc = correct_cnt.float() / total_cnt
            acc_val.append(acc)
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    epoch, batch_idx + 1, ave_loss, acc))
        writer.add_scalar('test/loss', ave_loss, epoch)
        writer.add_scalar('test/acc', np.mean(acc_val), epoch)


