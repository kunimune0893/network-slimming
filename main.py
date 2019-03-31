from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import binaryconnect

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--binarize', action='store_true', default=False,
                    help='Binary Connect')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='only test')
parser.add_argument('--evaluate-save', default=None, type=str, metavar='PATH',
                    help='weigth save path')
parser.add_argument('--refine-tmp', default=None, type=str, metavar='PATH',
                    help='path to the pruned model for cfg')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

cfg = []
if args.refine:
    checkpoint = torch.load(args.refine, map_location='cpu')
    if args.refine_tmp:
        ckpt = torch.load(args.refine_tmp, map_location='cpu')
        cfg = ckpt['cfg']
    else:
        cfg = checkpoint['cfg']
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=cfg)
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

bin_op = None
if args.binarize:
    bin_op = binaryconnect.BC( model )

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        raise

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1

def train( epoch, bin_op=None ):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        if bin_op is not None:
            bin_op.binarization()
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        _, pred = torch.max(output.data, 1)
        loss.backward()
        
        if args.sr:
            updateBN()
        
        if bin_op is not None:
            bin_op.restore()
        
        optimizer.step()
        
        if bin_op is not None:
            bin_op.clip()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test( epoch, bin_op=None ):
    model.eval()
    test_loss = 0
    correct = 0
    
    if bin_op is not None:
        bin_op.binarization()
    
    with torch.no_grad():
        for ii, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            _, pred = torch.max(output.data, 1) # get the index of the max log-probability
            correct += (pred == target).sum().item()
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            print( target )
            break
    
    if bin_op is not None:
        bin_op.restore()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def summary( model, savepath=None ):
    prnum = { "mod": 0, "conv": 0, "sel": 0, "bnw": 0, "bnb": 0, "fcw": 0, "fcb": 0, }
    
    for ii, (name, mm) in enumerate(model.named_modules()):
        print( "ii=", ii, ": name=", name, ", type=", type(mm), mm )
        
        if isinstance( mm, nn.Conv2d ):
            prnum['conv'] += mm.weight.reshape(-1).shape[0]
            prnum['mod']  += 1
            if savepath: np.save( os.path.join(savepath, name), mm.weight.data.numpy() )
        elif isinstance( mm, nn.BatchNorm2d ):
            prnum['bnw'] += mm.weight.shape[0]
            prnum['bnb'] += mm.bias.shape[0]
            prnum['mod'] += 2
            if savepath: np.save( os.path.join(savepath, name + ".w"), mm.weight.data.numpy() )
            if savepath: np.save( os.path.join(savepath, name + ".b"), mm.bias.data.numpy() )
            if savepath: np.save( os.path.join(savepath, name + ".m"), mm.running_mean.data.numpy() )
            if savepath: np.save( os.path.join(savepath, name + ".v"), mm.running_var.data.numpy() )
        elif isinstance( mm, models.channel_selection ):
            #print( mm.indexes )
            prnum['sel'] += mm.indexes.shape[0]
            prnum['mod'] += 1
            if savepath: np.save( os.path.join(savepath, name), mm.indexes.data.numpy() )
        elif isinstance( mm, nn.Linear ):
            prnum['fcw'] += mm.weight.reshape(-1).shape[0]
            prnum['fcb'] += mm.bias.shape[0]
            prnum['mod']  += 2
            if savepath: np.save( os.path.join(savepath, name + ".w"), mm.weight.data.numpy() )
            if savepath: np.save( os.path.join(savepath, name + ".b"), mm.bias.data.numpy() )

if args.evaluate_save:
    if bin_op is not None:
        bin_op.binarization()
    summary( model, args.evaluate_save )
    exit( 0 )

if args.evaluate:
    prec1 = test( epoch=0, bin_op=bin_op )
    exit( 0 )

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train( epoch, bin_op )
    prec1 = test( epoch, bin_op )
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        print( "best_prec1={:.3f}, prec1={:.3f}, is_best={}".format(best_prec1, prec1, is_best) )
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))
