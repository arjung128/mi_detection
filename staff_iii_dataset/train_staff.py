from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wfdb
import time
import random
from sklearn.preprocessing import minmax_scale
import sys
import pandas
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 16, 'batch_size')
flags.DEFINE_integer('gpu', 0, 'Which GPU to use.')
flags.DEFINE_integer('channel', 0, 'channel #1')
flags.DEFINE_integer('run', 0, 'run #')
flags.DEFINE_integer('seed', 0, 'seed #')

'''
TODO:
- identify best seed using multiple runs
   - reproduce original, no seed, results? clean up other training script and see what results it gives?
   - to combat overfitting for a fair comparison, try a shallower network?
- identify best channels using best seeds (multiple runs)

- pairwise, triplet-wise channels
- split by patient
'''


class StaffIIIDataset(Dataset):
    
    def __init__(self, record_path, excel_path, channel, train=True, length=10000):
        
        self.train = train
        self.length = length
        
        with open(record_path) as fp:  
            self.lines = fp.readlines()
            
        # shuffle with seed
        # np.random.seed(int(FLAGS.seed))
        # np.random.shuffle(self.lines)
                    
        if self.train:
            self.lines = self.lines[:int(0.8*len(self.lines))]
        else:
            self.lines = self.lines[int(0.8*len(self.lines)):]
            
        self.df = pandas.read_excel(excel_path)
        self.labels = self.df[u'Unnamed: 28'][8:].as_matrix()
        
        # channel
        self.channel = channel
        
        # bad_idx
        self.bad_idx = ['data/089d']
        
    def __getitem__(self, index):
        
        while True:
        
            # extract patient_id from file_name
            patient_id = int(self.lines[index][5:8])

            file_name = self.lines[index][:-1]
            # check file_name
            if file_name in self.bad_idx:
                index = np.random.randint(0, len(self.lines))
                continue
            
            data, _ = wfdb.rdsamp("staff-iii-database-1.0.0/" + str(file_name))
            data = np.array(data) # (300000, 9)

            # pick an arbitrary window
            start = random.choice(np.arange((data.shape[0]) - self.length)) 
            data = data[start:start+self.length, :]

            # extract relevant channels
            data = data[:, self.channel]

            if np.isnan(np.sum(data)):
                index = np.random.randint(0, len(self.lines))
            else:
                break
        
        # normalize
        data = minmax_scale(data)
        
        # unsqueeze
        data = data[np.newaxis, ...]

        if self.labels[patient_id] != 'no':
            y = 1
        else:
            y = 0
        
        return data, y
    
    def __len__(self):
        
        return len(self.lines)
        

        
class ConvNetQuake(nn.Module):
    def __init__(self):
        super(ConvNetQuake, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(1280, 128)
        self.linear2 = nn.Linear(128, 1)
        self.linear3 = nn.Linear(40000, 1)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(32)
        self.bn7 = nn.BatchNorm1d(32)
        self.bn8 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.bn1(F.relu((self.conv1(x))))
        x = self.bn2(F.relu((self.conv2(x))))
        x = self.bn3(F.relu((self.conv3(x))))
        '''
        x = self.bn4(F.relu((self.conv4(x))))
        x = self.bn5(F.relu((self.conv5(x))))
        x = self.bn6(F.relu((self.conv6(x))))
        x = self.bn7(F.relu((self.conv7(x))))
        x = self.bn8(F.relu((self.conv8(x))))
        '''
        x = torch.reshape(x, (FLAGS.batch_size, -1))
        '''
        x = self.linear1(x)
        x = self.linear2(x)
        '''
        x = self.linear3(x)
        x = self.sigmoid(x)     
        return x


def train(model, device, train_loader, optimizer, epoch, val_loader, writer, iteration):
    
    model.train()
    print_every = 10
    iteration_ = iteration
    criterion = nn.BCELoss()
    
    train_acc = 0
    train_loss = 0
   
    optimizer.zero_grad()
 
    for batch_idx, (data, y) in enumerate(train_loader):
        
        data, y = data.to(device).float(), y.to(device).float()
    
        y_pred = model(data)
        
        loss = criterion(y_pred, y)
        train_loss += loss.item()

        loss.backward()
        if (batch_idx+1)%(4) == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        pred = y_pred.round()
        train_acc += pred.eq(y.view_as(pred)).sum().item()
        
        if batch_idx%print_every == 0 and batch_idx != 0:
            
            iteration_ += 1
            
            train_loss /= (print_every + 1)
            train_acc /= float((print_every + 1) * FLAGS.batch_size)
            writer.add_scalar('Accuracy/train', train_acc, iteration_*print_every)
            writer.add_scalar('Loss/train', train_loss, iteration_*print_every)
            
            # validate
            val_acc, val_loss = validate(model, device, val_loader, 10) 
            model.train()
            writer.add_scalar('Accuracy/val', val_acc, iteration_*print_every)
            writer.add_scalar('Loss/val', val_loss, iteration_*print_every)
            
            np_file_path = 'staff_seed_results.npy'
            np_arr = np.load(np_file_path)
            np_arr[FLAGS.seed][FLAGS.run] = val_acc
            np.save(np_file_path, np_arr)
            
            # lr
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', lr, iteration_*print_every)

            train_loss = 0
            train_acc = 0
    
    return iteration_


def validate(model, device, val_loader, print_every):
    
    model.eval()
    criterion = nn.BCELoss()
    
    val_acc = 0
    val_loss = 0
    j = 0
    
    with torch.no_grad():
        for batch_idx, (data, y) in enumerate(val_loader):

            data, y = data.to(device).float(), y.to(device).float()

            y_pred = model(data)

            loss = criterion(y_pred, y)
            
            pred = y_pred.round()
            val_acc += pred.eq(y.view_as(pred)).sum().item()
        
            val_loss += loss.item()

            j += 1
            if j == print_every:
                break

    # here, num_batches = print_every, not print_every+1 as in train()
    val_acc /= float(print_every * FLAGS.batch_size)
    val_loss /= print_every

    return val_acc, val_loss
    
    
channels = {0: "v1", 
            1: "v2",
            2: "v3",
            3: "v4",
            4: "v5",
            5: "v6",
            6: "i",
            7: "ii",
            8: "iii"}

def main(argv):
        
    torch.cuda.set_device(FLAGS.gpu)

    # datasets
    train_dataset = StaffIIIDataset(record_path='staff-iii-database-1.0.0/RECORDS',
                                    excel_path='staff-iii-database-1.0.0/STAFF-III-Database-Annotations.xlsx',
                                    channel=FLAGS.channel)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=FLAGS.batch_size,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    val_dataset = StaffIIIDataset(record_path='staff-iii-database-1.0.0/RECORDS',
                                  excel_path='staff-iii-database-1.0.0/STAFF-III-Database-Annotations.xlsx',
                                  channel=FLAGS.channel,
                                  train=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=FLAGS.batch_size,
                            num_workers=1,
                            shuffle=False,
                            drop_last=True)
    
    device = torch.device("cuda")
    model = ConvNetQuake().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    scheduler = StepLR(optimizer, step_size=120, gamma=0.1)

    writer = SummaryWriter('/home/arjung2/mi_detection/staff_iii_dataset/runs_staff/runs_debug')
    # writer = SummaryWriter('/home/arjung2/mi_detection/staff_iii_dataset/runs_staff/runs_' + str(FLAGS.seed) + '_' + str(channels[FLAGS.channel]) + "_" + str(FLAGS.run))
    iteration = 0

    for epoch in range(1, 180):
        print("Train Epoch: ", epoch)
        iteration = train(model, device, train_loader, optimizer, epoch, val_loader, writer, iteration)
        scheduler.step()


if __name__ == '__main__':
    app.run(main)
        
