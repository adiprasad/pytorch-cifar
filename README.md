# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- Horovod with Pytorch support

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

## Multi GPU training using Horovod

### Code changes

This tutorial will take you step by step through the changes required in the existing training code (main.py) to run it across multiple GPUs using Horovod.   

The final script(main_horovod.py) with all the changes has been included in the repository. 

#### 1. Add Horovod import

Add the following code after `from utils import progress_bar`:

```python
import horovod.torch as hvd
```

![image](https://user-images.githubusercontent.com/8098496/128576792-14b64a1e-1a23-4503-8c05-474e6b034d04.png)
(see line 16)

#### 2. Initialize Horovod

Add the following code after `args = parser.parse_args()`:

```python
# Horovod: initialize Horovod.
hvd.init()
```

![image](https://user-images.githubusercontent.com/8098496/128576800-5b289cae-ebc2-4773-b87d-27d6bc43ebb5.png)
(see line 30-31)

#### 3. Pin GPU to be used by each process

With Horovod, usually one GPU is assigned per process to simplify distributed training across processes. 

Comment out or remove the following device/GPU allocation code

 ```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Add the following code after `hvd.init()`:

```python
## Get the rank of current process
device = hvd.local_rank()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(device)
```

![image](https://user-images.githubusercontent.com/8098496/128576806-dc6907af-6909-4371-9769-de79a4b96090.png)
(see line 33-37)

#### 4. Add distributed sampler for distributed sampling across processes

For distributed training, it is efficient to have each copy((on different processes)) of the model work with mutually exclusive subsamples of the training dataset. 

For this reason, we add a DistributedSampler to sample the training examples. Notice that we add the sampler as an argument to the DataLoader

Replace the following lines:- 

```python
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
```

by 

```python
# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    trainset, shuffle=True, num_replicas=hvd.size(), rank=hvd.rank())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, num_workers=2, sampler=train_sampler)
```

Similarly, we can distribute the evaluation load across processes during the validation phase

Replace the following lines:- 

```python
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
```

by 

```python
# Partition val dataset among workers using DistributedSampler
val_sampler = torch.utils.data.distributed.DistributedSampler(
    testset, shuffle=True, num_replicas=hvd.size(), rank=hvd.rank())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, sampler=val_sampler)
```

![image](https://user-images.githubusercontent.com/8098496/128576814-2f48d123-0335-4778-829f-902c9b010e7d.png)
(see line 60-73)

#### 5. Read checkpoint only on the first worker

Instead of loading the checkpoint from each worker process, it is more efficient to load the checkpoint through a single worker process(typically the root) and broadcast it to others. 

This is usually done in tandem with the checkpointing (Use single processes to store checkpoints)  

Replace the following code:

```python
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    ...
```

with:

```python
if args.resume and hvd.rank() == 0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
```

![image](https://user-images.githubusercontent.com/8098496/128576822-61008780-aa1f-4c21-b620-6b0a19e4683a.png)
(see line 102)


#### 6. Broadcast start epoch and model parameters from first worker to all processes

As mentioned in the previous section, the checkpoint and model parameters are broadcast(from the root process) and synchronized with other processes. 

Add the following lines of code after the checkpoint reading code :- 

```python
start_epoch = hvd.broadcast(torch.Tensor(1).fill_(start_epoch)[0], name="start_epoch", root_rank=0)
start_epoch = int(start_epoch)
```

Also add :- 

```python
hvd.broadcast_parameters(net.state_dict(), 0)
```

![image](https://user-images.githubusercontent.com/8098496/128578475-88df348c-7741-47cb-98d6-cc8e0636f27f.png)
(see line 111-114)

#### 7. Adjust learning rate and add Distributed Optimizer

Horovod uses an operation that averages gradients across workers.  Gradient averaging typically requires a corresponding increase in learning rate to make bigger steps in the direction of a higher-quality gradient.

Replace `optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)` with:

```python
optimizer = optim.SGD(net.parameters(), lr=args.lr * hvd.size(),
                      momentum=0.9, weight_decay=5e-4)

## Add distributed optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
```

![image](https://user-images.githubusercontent.com/8098496/128576845-284da662-3bce-47ec-b218-8912602bb440.png)
(see line 117-121)

#### 8. Broadcast optimizer state from first worker to synchronize the optimizer across processes

Add the following line after `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)`

```python
hvd.broadcast_optimizer_state(optimizer, 0)
```

![image](https://user-images.githubusercontent.com/8098496/128576848-df67ff27-a987-45e9-99dd-0ee13c371961.png)
(see line 125)

#### 9. Aggregate losses and predictions across processes to calculate overall loss and accuracy

As mentioned in Section 4, every process works on their own subsample of the training set. We don't want every process to report their own training progress but rather have a single process report the aggregate training metrics.

For reporting aggregate metrics, we need to average them across all the processes.  

Add the following lines after `correct += predicted.eq(targets).sum().item()`

```python
train_loss_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(train_loss)[0],
                                                           name="train_loss_sum_multiprocess", op=Sum)
total_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(total)[0],
                                                      name="total_sum_multiprocess", op=Sum)
correct_sum_across_batches_multiprocess = hvd.allreduce(torch.Tensor(1).fill_(correct)[0],
                                                        name="correct_sum_multiprocess", op=Sum)
``` 

These lines are summing up the per process loss, number of total examples and number of correct examples respectively and storing them into new variables, which will later be used for reporting.   

To enable reports from a single(root) process only, replace the following code :- 

```python
progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
```

with 

```python
    if hvd.local_rank() == 0:
        progress_bar(batch_idx, len(trainloader), '[Train] Average(all procs) Loss : %.3f | Average(all procs) Acc: %.3f%% (%d/%d)'
                % (train_loss_sum_across_batches_multiprocess/((batch_idx+1) * hvd.size()),
                100.*correct_sum_across_batches_multiprocess/total_sum_across_batches_multiprocess,
                correct_sum_across_batches_multiprocess, total_sum_across_batches_multiprocess))
```

![image](https://user-images.githubusercontent.com/8098496/128576855-8ff4cb87-3bb0-4f51-8a73-c2abd80b0ebb.png)
(see line 148-159)


Note that similar aggregation is also performed for validation (See line 179-190 inside ***test*** function)


### Run

Assuming the libraries mentioned as pre-requisites are installed in your python environment :- 

```
horovodrun -np <num-gpus> python main_horovod.py <args>
``` 