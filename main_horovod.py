'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import horovod.torch as hvd

from horovod.torch.mpi_ops import Sum


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ckpt-dir', default="distributed_checkpoint", type=str, help='checkpoint directory')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Horovod
hvd.init()

## Get the rank of current process
device = hvd.local_rank()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    trainset, shuffle=True, num_replicas=hvd.size(), rank=hvd.rank())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, num_workers=2, sampler=train_sampler)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# Partition val dataset among workers using DistributedSampler
val_sampler = torch.utils.data.distributed.DistributedSampler(
    testset, shuffle=True, num_replicas=hvd.size(), rank=hvd.rank())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, sampler=val_sampler)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = ResNet50()
net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume and hvd.rank() == 0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.ckpt_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./{}/ckpt.pth'.format(args.ckpt_dir))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

start_epoch = hvd.broadcast(torch.Tensor(1).fill_(start_epoch)[0], name="start_epoch", root_rank=0)
start_epoch = int(start_epoch)

hvd.broadcast_parameters(net.state_dict(), 0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr * hvd.size(),
                      momentum=0.9, weight_decay=5e-4)

## Add distributed optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

hvd.broadcast_optimizer_state(optimizer, 0)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(train_loss)[0],
                                                                   name="train_loss_sum_multiprocess", op=Sum)
        total_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(total)[0],
                                                              name="total_sum_multiprocess", op=Sum)
        correct_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(correct)[0],
                                                                name="correct_sum_multiprocess", op=Sum)

        if hvd.local_rank() == 0:
            progress_bar(batch_idx, len(trainloader), '[Train] Average(all procs) Loss : %.3f | Average(all procs) Acc: %.3f%% (%d/%d)'
                        % (train_loss_sum_across_batches_multiprocess/((batch_idx+1) * hvd.size()),
                           100.*correct_sum_across_batches_multiprocess/total_sum_across_batches_multiprocess,
                           correct_sum_across_batches_multiprocess, total_sum_across_batches_multiprocess))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            test_loss_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(test_loss)[0],
                                                                       name="test_loss_sum_multiprocess", op=Sum)
            test_total_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(total)[0],
                                                                  name="test_total_sum_multiprocess", op=Sum)
            test_correct_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(correct)[0],
                                                                    name="test_correct_sum_multiprocess", op=Sum)

            if hvd.local_rank() == 0:
                progress_bar(batch_idx, len(testloader), '[Val] Average(all procs) Loss : %.3f | Average(all procs) Acc: %.3f%% (%d/%d)'
                            % (test_loss_sum_across_batches_multiprocess/((batch_idx+1)* hvd.size()),
                               100.*test_correct_sum_across_batches_multiprocess/test_total_sum_across_batches_multiprocess,
                               test_correct_sum_across_batches_multiprocess, test_total_sum_across_batches_multiprocess))

    # Save checkpoint.
    acc = 100.*test_correct_sum_across_batches_multiprocess/test_total_sum_across_batches_multiprocess
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if hvd.rank() == 0:
            if not os.path.isdir(args.ckpt_dir):
                os.mkdir(args.ckpt_dir)
            torch.save(state, './{}/ckpt.pth'.format(args.ckpt_dir))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
