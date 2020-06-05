from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
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

# channel_1 = sys.argv[1]
# channel_2 = sys.argv[2]
# seed_num = 37
channel_1 = "i"
writer = SummaryWriter('/home/arjung2/mi_detection/runs_staff/runs_staff_new_'+ str(channel_1))

print(channel_1)


# load real data (ptbdb)
# with open('ptbdb_data/RECORDS') as fp:  
#     lines = fp.readlines()
with open('staff-iii-database-1.0.0/RECORDS') as fp:  
    lines = fp.readlines()
    
lines = lines[:int(0.7*len(lines))]
    
df = pandas.read_excel('staff-iii-database-1.0.0/STAFF-III-Database-Annotations.xlsx')
labels = df[u'Unnamed: 28'][8:].as_matrix()

files_unhealthy, files_healthy = [], []

for file_ in lines:
        
    patient_id = int(file_[5:8])
        
    # print(file_[:-1], patient_id, labels[patient_id], labels[patient_id] == "no")
            
    if labels[patient_id] == "no":
        files_healthy.append(file_)
    else:
        files_unhealthy.append(file_)

# shuffle data (cross-validation)
# np.random.seed(int(seed_num))
np.random.shuffle(files_unhealthy)
np.random.shuffle(files_healthy)
        
data_unhealthy = []
for file in files_unhealthy:
    data_v4, _ = wfdb.rdsamp("staff-iii-database-1.0.0/" + file[:-1])
    # data = [data_v4.flatten()]
    data = (data_v4[:, 6]).flatten()
    data_unhealthy.append(data)
data_healthy = []
for file in files_healthy:
    data_v4, _ = wfdb.rdsamp("staff-iii-database-1.0.0/" + file[:-1])
    # data = [data_v4.flatten()]
    data = (data_v4[:, 6]).flatten()
    data_healthy.append(data)
    
'''

# load real data (ptbdb)
with open('ptbdb_data/RECORDS') as fp:  
    lines = fp.readlines()

files_unhealthy, files_healthy = [], []

for file in lines:
    file_path = "ptbdb_data/" + file[:-1] + ".hea"
    
    # read header to determine class
    if 'Myocardial infarction' in open(file_path).read():
        files_unhealthy.append(file)
        
    if 'Healthy control' in open(file_path).read():
        files_healthy.append(file)

# shuffle data (cross-validation)
# np.random.seed(int(seed_num))
np.random.shuffle(files_unhealthy)
np.random.shuffle(files_healthy)
        
data_unhealthy = []
for file in files_unhealthy:
    data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str('i')])
    data = data_v4.flatten()
    data_unhealthy.append(data)
data_healthy = []
for file in files_healthy:
    data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str('i')])
    data = data_v4.flatten()
    data_healthy.append(data)
    
'''

data_unhealthy = np.asarray(data_unhealthy)
data_healthy = np.asarray(data_healthy)

num_unhealthy = (data_unhealthy.shape)[0]
num_healthy = (data_healthy.shape)[0]

window_size = 10000

def get_batch(batch_size, split='train'):
    
    unhealthy_threshold = int(0.8*num_unhealthy)
    healthy_threshold = int(0.8*num_healthy)
    
    unhealthy_test_threshold = int(0.9*num_unhealthy)
    healthy_test_threshold = int(0.9*num_healthy)
    
    if split == 'train':
        unhealthy_indices = random.sample(np.arange(unhealthy_threshold), k=int(batch_size / 2))
        healthy_indices = random.sample(np.arange(healthy_threshold), k=int(batch_size / 2))
    elif split == 'val': 
        unhealthy_indices = random.sample(unhealthy_threshold + np.arange(unhealthy_test_threshold - unhealthy_threshold), k=int(batch_size / 2))
        healthy_indices = random.sample(healthy_threshold + np.arange(healthy_test_threshold - healthy_threshold), k=int(batch_size / 2))
    elif split == 'test':
        unhealthy_indices = random.sample(unhealthy_test_threshold + np.arange(num_unhealthy - unhealthy_test_threshold), k=int(batch_size / 2))
        healthy_indices = random.sample(healthy_test_threshold + np.arange(num_healthy - healthy_test_threshold), k=int(batch_size / 2))
            
    unhealthy_batch = data_unhealthy[unhealthy_indices]
    healthy_batch = data_healthy[healthy_indices]
    
    batch_x = []
    for sample in unhealthy_batch:
        
        start = random.choice(np.arange(len(sample) - window_size))
        
        # normalize
        normalized_1 = minmax_scale(sample[start:start+window_size])
        normalized = np.array(normalized_1)
        
        batch_x.append(normalized)
        
    for sample in healthy_batch:
        
        start = random.choice(np.arange(len(sample) - window_size))
        
        # normalize
        normalized_1 = minmax_scale(sample[start:start+window_size])
        normalized = np.array(normalized_1)
        
        batch_x.append(normalized)
    
    batch_y = [0 for _ in range(int(batch_size / 2))]
    for _ in range(int(batch_size / 2)):
        batch_y.append(1)
        
    indices = np.arange(len(batch_y))
    np.random.shuffle(indices)
    
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    batch_x = batch_x[indices]
    batch_y = batch_y[indices]
    
    batch_x = np.reshape(batch_x, (-1, 1, window_size))
    batch_x = torch.from_numpy(batch_x)
    batch_x = batch_x.float().cuda()
    batch_x = batch_x.float()
    
    batch_y = np.reshape(batch_y, (-1, 1))
    batch_y = torch.from_numpy(batch_y)
    batch_y = batch_y.float().cuda()
    batch_y = batch_y.float()
    
    return batch_x, batch_y


# model

class ConvNetQuake(nn.Module):
    def __init__(self):
        super(ConvNetQuake, self).__init__()
        
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
#         self.linear1 = nn.Linear(1280, 128)
#         self.linear2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.bn4 = nn.BatchNorm1d(32)
#         self.bn5 = nn.BatchNorm1d(32)
#         self.bn6 = nn.BatchNorm1d(32)
#         self.bn7 = nn.BatchNorm1d(32)
#         self.bn8 = nn.BatchNorm1d(32)

        # logistic regression
#         self.linear1 = nn.Linear(10000, 1)
#         self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(40000, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
#         x = self.bn1(F.relu((self.conv1(x))))
#         x = self.bn2(F.relu((self.conv2(x))))
#         x = self.bn3(F.relu((self.conv3(x))))
#         x = self.bn4(F.relu((self.conv4(x))))
#         x = self.bn5(F.relu((self.conv5(x))))
#         x = self.bn6(F.relu((self.conv6(x))))
#         x = self.bn7(F.relu((self.conv7(x))))
#         x = self.bn8(F.relu((self.conv8(x))))
#         x = torch.reshape(x, (10, -1))
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.sigmoid(x)

#         x = self.linear1(x)
#         x = self.sigmoid(x)
        
        if torch.any(torch.isnan(x)):
            x = torch.zeros(10, 1, 10000).cuda()
            print("nan")
    
        input_ = x
        x = self.bn1(F.relu((self.conv1(x))))
        e = x
        x = self.bn2(F.relu((self.conv2(x))))
        d = x
        x = self.bn3(F.relu((self.conv3(x))))
        c = x
        x = torch.reshape(x, (10, -1))
        b = x
        x = self.linear1(x)
        a = x
        x = self.sigmoid(x)
        
#         if torch.any(torch.isnan(x)):
#             print("x: ", x)
#             print("a: ", a)
#             print("b: ", b)
#             print("c: ", c)
#             print("d: ", d)
#             print("e: ", e)
#             print(torch.any(torch.isnan(input_)))
#             x = torch.zeros(10).cuda()

        return x
    

# define model
model = ConvNetQuake()
model.cuda()

model = nn.DataParallel(model, device_ids=[0])

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-6)
criterion = nn.BCELoss()

# training loop

num_iters = 150000
batch_size = 10

acc_values = []
acc_values_train = []

for iters in range(num_iters):

    batch_x, batch_y = get_batch(batch_size, split='train')

    y_pred = model(batch_x)

    y_pred = y_pred.squeeze()
    batch_y = batch_y.squeeze()
    # print(y_pred)
    loss = criterion(y_pred, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    if iters%100 == 0 and iters != 0:
        
        print("in val")
        writer.add_scalar('Loss/train', loss, iters/100)

        with torch.no_grad():

            # test_set
            iterations = 100
            avg_acc = 0

            for _ in range(iterations):
                batch_x, batch_y = get_batch(batch_size, split='val')
                cleaned = model(batch_x)

                count = 0
                acc = 0
                for num in cleaned:
                    if int(round(num)) == int(round(batch_y[count])):
                        acc += 10
                    count += 1
                avg_acc += acc

            acc_values.append((avg_acc / iterations))
            writer.add_scalar('Accuracy/val', (avg_acc / iterations), iters/100)

            # train_set
            iterations = 100
            avg_acc = 0

            for _ in range(iterations):
                batch_x, batch_y = get_batch(batch_size, split='train')
                cleaned = model(batch_x)

                count = 0
                acc = 0
                for num in cleaned:
                    if int(round(num)) == int(round(batch_y[count])):
                        acc += 10
                    count += 1
                avg_acc += acc

            acc_values_train.append((avg_acc / iterations))
            writer.add_scalar('Accuracy/train', (avg_acc / iterations), iters/100)

#     if iters%1000 == 0 and iters != 0:

#         torch.save(model.state_dict(), 'CNQ_model.pth')
#         torch.save(optimizer.state_dict(), 'CNQ_optim.opt')

#         fig = plt.figure(figsize=(18, 12))
#         plt.title(iters)
#         plt.plot(acc_values, color="blue")
#         plt.plot(acc_values_train, color="red")
#         plt.grid()
#         fig.savefig("CNQ_model.jpeg")

plt.close()

with torch.no_grad():

    # test_set
    iterations = 1000
    avg_acc = 0

    for _ in range(iterations):
        batch_x, batch_y = get_batch(batch_size, split='test')
        cleaned = model(batch_x)

        count = 0
        acc = 0
        for num in cleaned:
            if int(round(num)) == int(round(batch_y[count])):
                acc += 10
            count += 1
        avg_acc += acc

    print(float(avg_acc) / iterations)

del model

