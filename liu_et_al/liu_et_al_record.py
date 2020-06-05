'''
Changes:
- downsample to 250 Hz -- DONE
- 3 second long inputs -- DONE
- remove label smoothing -- DONE
- channels (lead ii) -- DONE
- batch size -- DONE
- architecture -- DONE
- l2 reg on last linear layer -- DONE
- learning rate decay -- DONE
- stopping criterion -- DONE
'''

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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import FileWriter
from torch.optim.lr_scheduler import StepLR

channel_1 = 'ii'
# channel_2 = sys.argv[2]
seed_num = 33
run_num = 6

print(seed_num, channel_1)

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
np.random.seed(int(seed_num))
np.random.shuffle(files_unhealthy)
np.random.shuffle(files_healthy)
        
data_unhealthy = []
for file in files_unhealthy:
    data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_1)])
    data = [data_v4.flatten()]
    data_unhealthy.append(data)
data_healthy = []
for file in files_healthy:
    data_v4, _ = wfdb.rdsamp("ptbdb_data/" + file[:-1], channel_names=[str(channel_1)])
    data = [data_v4.flatten()]
    data_healthy.append(data)

data_unhealthy = np.asarray(data_unhealthy)
data_healthy = np.asarray(data_healthy)

num_unhealthy = (data_unhealthy.shape)[0]
num_healthy = (data_healthy.shape)[0]

window_size = 3000

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
        
        start = random.choice(np.arange(len(sample[0]) - window_size))
        
        # normalize
        normalized = minmax_scale(sample[0][start:start+window_size])

        # downsample
        normalized = normalized[0:normalized.size:4]
        
        batch_x.append(normalized)
        
    for sample in healthy_batch:
        
        start = random.choice(np.arange(len(sample[0]) - window_size))
        
        # normalize
        normalized = minmax_scale(sample[0][start:start+window_size])
        
        # downsample
        normalized = normalized[0:normalized.size:4]
        
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
    
    batch_x = np.reshape(batch_x, (-1, 1, 750))
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
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=151, stride=1)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=45, stride=1)
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=20, stride=1)
        self.conv4 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=10, stride=1)
        self.linear1 = nn.Linear(320, 30)
        self.linear2 = nn.Linear(30, 10)
        self.linear3 = nn.Linear(10, 2)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(3)
        self.bn2 = nn.BatchNorm1d(10)
        self.bn3 = nn.BatchNorm1d(10)
        self.bn4 = nn.BatchNorm1d(10)
        self.mp1 = nn.MaxPool1d(6, stride=2, padding=2)
        self.mp2 = nn.MaxPool1d(20, stride=2, padding=9)
        self.mp3 = nn.MaxPool1d(20, stride=2, padding=9)
        self.mp4 = nn.MaxPool1d(20, stride=2, padding=9)
        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = F.relu(self.bn1((self.conv1(x))))
        x = self.mp1(x)
        x = F.relu(self.bn2((self.conv2(x))))
        x = self.mp2(x)
        x = F.pad(self.conv3(x), (9, 10), "constant", 0)
        x = F.relu(self.bn3(x))
        x = self.mp3(x)
        x = F.pad(self.conv4(x), (4, 5), "constant", 0)
        x = F.relu(self.bn4(x))
        x = self.mp4(x)
        x = self.drop1(x)
        x = torch.reshape(x, (batch_size, -1))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.drop2(x)
        x = self.linear3(x)

        return x
    

# define model
model = ConvNetQuake()
model.cuda()

# model = nn.DataParallel(model, device_ids=[0])
device = torch.device("cuda:0")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter('/home/arjung2/mi_detection/runs_liu_et_al/runs_record_' + str(seed_num) + '_' + str(run_num) + '_' + str(channel_1))

# training loop
num_iters = 45000
batch_size = 10

acc_values = []
acc_values_train = []

for iters in range(num_iters):

    batch_x, batch_y = get_batch(batch_size, split='train')

    y_pred = model(batch_x)

    batch_y = batch_y.type(torch.LongTensor).cuda().squeeze()
    loss = criterion(y_pred, batch_y)
    
    # L2 regularization
    l1_crit = nn.L1Loss(size_average=False)
    reg_loss = 0
    for param in model.linear3.parameters():
        # reg_loss += l1_crit(param)
        reg_loss += F.l1_loss(param, target=torch.zeros_like(param), size_average=False)
    factor = 0.001
    loss += factor * reg_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation
    if iters%100 == 0 and iters != 0:
        
        writer.add_scalar('Loss/train', loss, iters)

        with torch.no_grad():

            # test_set
            iterations = 100
            avg_acc = 0

            for _ in range(iterations):
                batch_x, batch_y = get_batch(batch_size, split='val')
                cleaned = model(batch_x)
                
                batch_x = batch_x[:10]
                batch_y = batch_y[:10]
                cleaned = cleaned[:10]

                count = 0
                acc = 0
                for num in cleaned:
                    if int(torch.argmax(num)) == int(round(batch_y[count])):
                        acc += 10
                    count += 1
                avg_acc += acc

            acc_values.append((avg_acc / iterations))
            writer.add_scalar('Accuracy/val', (avg_acc / iterations), iters)

            # train_set
            iterations = 100
            avg_acc = 0

            for _ in range(iterations):
                batch_x, batch_y = get_batch(batch_size, split='train')
                cleaned = model(batch_x)
                
                batch_x = batch_x[:10]
                batch_y = batch_y[:10]
                cleaned = cleaned[:10]

                count = 0
                acc = 0
                for num in cleaned:
                    if int(torch.argmax(num)) == int(round(batch_y[count])):
                        acc += 10
                    count += 1
                avg_acc += acc

            acc_values_train.append((avg_acc / iterations))
            writer.add_scalar('Accuracy/train', (avg_acc / iterations), iters)
            
    if iters%10000 == 0 and iters != 0:
        scheduler.step()

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

# with torch.no_grad():

#     # test_set
#     iterations = 1000
#     avg_acc = 0

#     for _ in range(iterations):
#         batch_x, batch_y = get_batch(batch_size, split='test')
#         cleaned = model(batch_x)

#         count = 0
#         acc = 0
#         for num in cleaned:
#             if int(round(num)) == int(round(batch_y[count])):
#                 acc += 10
#             count += 1
#         avg_acc += acc

#     print(float(avg_acc) / iterations)

del model

