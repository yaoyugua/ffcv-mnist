from __future__ import print_function
import time
start = time.time()

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

class Argument():
    def __init__(self, batch_size=256, test_batch_size=2000, epochs=1, lr=1e-2, no_cuda=False, save_model=False):
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.no_cuda = no_cuda
        self.save_model = save_model

args = Argument()
use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

train_data = torch.load('data/train_data.pt', map_location=device)
train_labels = torch.load('data/train_labels.pt', map_location=device)

test_data = torch.load('data/test_data.pt', map_location=device)
test_labels = torch.load('data/test_labels.pt', map_location=device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 5, 1)
        self.conv2 = nn.Conv2d(24, 32, 3, 1)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, optimizer, scheduler, epochs):
    batch_size = args.batch_size
    model.train()
    batches_per_epoch = (len(train_data)-1)//batch_size + 1
    
    for i in range(int(epochs*batches_per_epoch)):
        i = i%batches_per_epoch
        data = train_data[batch_size*i:batch_size*(i+1)]
        target = train_labels[batch_size*i:batch_size*(i+1)]
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
    


def test(args, model, device, epoch):
    model.eval()
    batch_size = args.test_batch_size
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i in range(len(test_data)//batch_size):
            data = test_data[batch_size*i:batch_size*(i+1)]
            target = test_labels[batch_size*i:batch_size*(i+1)]
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_labels)

    #print('Epoch {} Test set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
    #    epoch, test_loss, 100. * correct / len(test_data)))
    return 100.*correct / len(test_labels)


def main():
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.7, 0.9))

    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=int(((len(train_data)-1)//args.batch_size + 1)*args.epochs), 
                           cycle_momentum=False)
    
    train(args, model, device, optimizer, scheduler, args.epochs)
        
    test_acc = test(args, model, device, args.epochs+1)
    print(test_acc, '%')
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    return test_acc


main()
print(time.time() - start, 's')
