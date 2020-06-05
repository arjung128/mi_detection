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

# In[2]:


class StaffIIIDataset(Dataset):
    
    def __init__(self, record_path, excel_path, channel, train=True, length=10000, transform=None):
        
        self.train = train
        self.length = length
        
        with open(record_path) as fp:  
            self.lines = fp.readlines()
                    
        if self.train:
            self.lines = self.lines[:int(0.8*len(self.lines))]
        else:
            self.lines = self.lines[int(0.8*len(self.lines)):]
            
        self.df = pandas.read_excel(excel_path)
        self.labels = self.df[u'Unnamed: 28'][8:].as_matrix()
        
        # channel
        self.channel = channel
        
    def __getitem__(self, index):
        
        # extract patient_id from file_name
        patient_id = int(self.lines[index][5:8])
        
        if patient_id == 28 or patient_id == 67:
            # train
            index = 0
            patient_id = int(self.lines[index][5:8])
        if  patient_id == 78 or patient_id == 103:
            # val
            index = 105
            patient_id = int(self.lines[index][5:8])
        
        file_name = self.lines[index][:-1]
        data, _ = wfdb.rdsamp("staff-iii-database-1.0.0/" + str(file_name))
        data = np.array(data) # (300000, 9)
        
        if np.isnan(np.sum(data)):
            print(patient_id)
        
        # pick an arbitrary window
        start = random.choice(np.arange((data.shape[0]) - self.length)) 
        data = data[start:start+self.length, :]
        
        # extract relevant channels
        data = data[:, self.channel]

        # normalize
        data = minmax_scale(data)
        
        # unsqueeze
        # data = torch.unsqueeze(data, 1) 
        data = data[np.newaxis, ...]

        if self.labels[patient_id] != 'no':
            y = 1
        else:
            y = 0
        
        return data, y
    
    def __len__(self):
        
        if self.train:
            return int(0.7*519)
        else:
            return int(0.1*519)
        


# In[8]:

batch_size = 10

# In[5]:


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
        x = self.bn4(F.relu((self.conv4(x))))
        x = self.bn5(F.relu((self.conv5(x))))
        x = self.bn6(F.relu((self.conv6(x))))
        x = self.bn7(F.relu((self.conv7(x))))
        x = self.bn8(F.relu((self.conv8(x))))
        x = torch.reshape(x, (batch_size, -1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)     
        return x


# In[17]:


def train(model, device, train_loader, optimizer, epoch, val_loader, writer, iteration):
    
    model.train()
    print_every = 10
    iteration_ = iteration
   
    optimizer.zero_grad()
 
    for batch_idx, (data, y) in enumerate(train_loader):
        
        data, y = data.to(device), y.to(device)
        data = data.cuda()
        y = y.cuda()
        data = data.float()        
        y = y.float()

        # optimizer.zero_grad()
    
        y_pred = model(data)
        
        loss = criterion(y_pred, y)

        loss.backward()
        # optimizer.step()
        if (batch_idx+1)%(4) == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if batch_idx%print_every == 0 and batch_idx != 0:
            
            iteration_ += 1
            writer.add_scalar('Loss/train', loss, iteration_)
            
            # validate
            with torch.no_grad():

                # test_set
                iters = 0
                avg_acc = 0
                for batch_idx, (data_val, y_val) in enumerate(val_loader):
                    
                    data_val, y_val = data_val.to(device), y_val.to(device)
                    data_val = data_val.float()
                    y_val = y_val.float()
                        
                    y_pred_val = model(data_val)

                    count = 0
                    acc = 0
                    for num in y_pred_val:
                        if int(round(num)) == int(round(y_val[count])):
                            acc += 10
                        count += 1
                    avg_acc += acc                

                    iters += 1
                    if iters == 5:
                        break
                
                writer.add_scalar('Accuracy/val', (avg_acc / 5), iteration_)

                # train_set
                iters = 0
                avg_acc = 0
                for batch_idx, (data_val, y_val) in enumerate(train_loader):

                    data_val, y_val = data_val.to(device), y_val.to(device)
                    data_val = data_val.float()
                    y_val = y_val.float()

                    y_pred_val = model(data_val)

                    count = 0
                    acc = 0
                    for num in y_pred_val:
                        if int(round(num)) == int(round(y_val[count])):
                            acc += 10
                        count += 1
                    avg_acc += acc

                    iters += 1
                    if iters == 5:
                        break
                
                writer.add_scalar('Accuracy/train', (avg_acc / 5), iteration_)
    
    return iteration_


# In[18]:


channels = {0: "v1", 
            1: "v2",
            2: "v3",
            3: "v4",
            4: "v5",
            5: "v6",
            6: "i",
            7: "ii",
            8: "iii"}

# channel_1 = 0

for channel_1 in range(9):

    train_dataset = StaffIIIDataset(record_path='staff-iii-database-1.0.0/RECORDS',
                                    excel_path='staff-iii-database-1.0.0/STAFF-III-Database-Annotations.xlsx',
                                    channel=channel_1)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)

    val_dataset = StaffIIIDataset(record_path='staff-iii-database-1.0.0/RECORDS',
                                  excel_path='staff-iii-database-1.0.0/STAFF-III-Database-Annotations.xlsx',
                                  channel=channel_1,
                                  train=False)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=1,
                            shuffle=False,
                            drop_last=True)

    model = ConvNetQuake()
    model.cuda()
    device = torch.device("cuda")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    criterion = nn.BCELoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter('/home/arjung2/mi_detection/staff_iii_dataset/runs_staff/runs_debugging_'+ str(channels[channel_1]))
    iteration = 0

    for epoch in range(1, 30):
        print("Train Epoch: ", epoch)
        iteration = train(model, device, train_loader, optimizer, epoch, val_loader, writer, iteration)
        scheduler.step()

# In[ ]:




