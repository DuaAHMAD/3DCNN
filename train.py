from __future__ import print_function
import argparse
import os
import random
from os.path import join
import torch
import torch.optim as optim
from torch.utils import data as Data
import shutil
import numpy as np

from learning.datasets import BDIDerivedDataset
from learning.models import BDI_3D_Conv, BDI_3D_Conv_Simple
from learning.loops import train, validate, test
import config
import learning.dataset_config as learning_config
# from tensorboard_logger import Logger
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='BDI Training Script')
parser.add_argument('--dataroot', default=config.ALIGNED_FACES_FOLDER, type=str, help='path to dataset')
parser.add_argument('--datasets', type=str, help='datasets used for training and validation')
parser.add_argument('--workers', '-j', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch', type=int, default=75, help='input batch size')
parser.add_argument('--epochs', default=25, type=int, help='number of epochs to run')
parser.add_argument('--cpu', action='store_true', help='run without cuda')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--checkpoint', type=str, help='location of the checkpoint to load')
parser.add_argument('--output', default='/home/mohammad/output/emotion', type=str,
                    help='folder to output model checkpoints')

parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--visualize', action='store_true', help='evaluate model on validation set')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--submedian', action='store_true', help='run on natural data')
parser.add_argument('--overwrite', action='store_true', help='overwrite the prediction data or not')
parser.add_argument('--augment', action='store_true', help='do data augmentation or not')

#parser.set_defaults(augment=True)

# parse arguments
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(1, 10000)

if args.visualize:
    # evaluation model
    args.output = args.checkpoint[0:-4]

# print arguments
print("Summary of Arguments:")
for key, val in vars(args).items():
    print("{:10} {}".format(key, val))


# handle random seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(int(args.seed))
if not args.cpu:
    torch.cuda.manual_seed_all(int(args.seed))

# create output folder
if os.path.exists(args.output):
    if not args.visualize and args.checkpoint == None:
        if len([f for f in os.listdir(args.output) if "model" in f]) > 0:
            raise(RuntimeError("Output folder {} already exist.".format(args.output)))
        else:
            shutil.rmtree(args.output)
            os.makedirs(args.output)
else:
    os.makedirs(args.output)

# create tensorboard, args and code copy
if not args.visualize:
#     train_logdir=join(args.output, "train")
#     val_logdir=join(args.output, "val")                  
    Logger_train = SummaryWriter(flush_secs=2)
    Logger_val = SummaryWriter(flush_secs=2)
    f_log = open(os.path.join(args.output, 'args.txt'), 'w')
    for key, val in vars(args).items():
        print("{} {}".format(key, val), file=f_log)
    f_log.close()
    if args.checkpoint is None:
        shutil.copytree('.', os.path.join(args.output, 'src'))

# specify train and validate data folders

train_exp, validate_exp = learning_config.get_train_val_folders(name=args.datasets)

# #TODO
# if args.visualize:
#     validate_exp += train_exp[0:2]
#     validate_exp += ["exp0718-2-01", "exp0718-2-02", "exp0719-2-01", "exp0719-2-02"]

# build datasets
train_dataset = BDIDerivedDataset(
    folders=[args.dataroot.format(exp=exp) for exp in train_exp],
    submedian=args.submedian,
    flip=args.augment
)
validate_dataset = BDIDerivedDataset(
    folders=[args.dataroot.format(exp=exp) for exp in validate_exp],
    submedian=args.submedian,
    flip=False,
    return_idx=args.visualize
)

# build data loaders
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=False
)
validate_loader = Data.DataLoader(
    dataset=validate_dataset,
    batch_size=args.batch,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=False
)

# print(len(train_loader))
# print(len(validate_loader))
#    
# for idx, data in enumerate(train_loader):
#     (rf, labels) = data
# for idx, data in enumerate(validate_loader):
#     (rf, labels) = data
# #     print(labels)
# #     print(np.array(rf).shape)
# exit()


# build model and optimizer
model = BDI_3D_Conv_Simple()
# model = BDI_3D_Conv()
# Device configuration
device = torch.device('cuda:0' if not args.cpu else 'cpu')
model.to(device)


print('# of params:', str(sum([p.numel() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss().cuda()

# optionally load model from a checkpoint
if args.checkpoint:
    if os.path.isfile(args.checkpoint):
        model = torch.load(args.checkpoint)
        if not args.visualize:
            start_epoch = int(args.checkpoint.split('.')[-2].split('_')[-1]) + 1
    else:
        raise(RuntimeError("no checkpoint found at '{}'".format(args.checkpoint)))
else:
    start_epoch = 1

lr = args.lr

if args.visualize:
    # evaluation model
    if not args.checkpoint:
        raise(RuntimeWarning("visualizing a random initialized model"))
    if args.overwrite:
        shutil.rmtree(args.output)
    test(model, validate_loader, 0, args, criterion=criterion)
else:
    # train and validate loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, args, criterion=criterion)
        val_loss = validate(model, validate_loader, epoch, args, criterion=criterion)

        # do tensorboard and checkpointing:
        Logger_train.log_value('loss', train_loss, epoch)
        Logger_val.log_value('loss', val_loss[0], epoch)
        Logger_val.log_value('accuracy', val_loss[1], epoch)
        Logger_train.log_value('lr', lr, epoch)
        torch.save(model, '{}/model_epoch_{}.pth'.format(args.output, epoch))
