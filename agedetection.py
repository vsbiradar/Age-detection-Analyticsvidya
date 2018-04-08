import numpy as np
import pandas as pd
import os
from PIL import Image

train = pd.read_csv('train.csv')
xtrain = np.zeros((train.shape[0],128,128,3))
for i in range(train.shape[0]):
    image = Image.open('Train'+'/'+train['ID'][i])
    image = image.resize((128,128))
    image = np.array(image)
    xtrain[i] = image
    
classes = {'YOUNG':0, 'MIDDLE':1, 'OLD':2}
y_train = train['Class']

ytrain = []
for i in range(train.shape[0]):
    ytrain.append(classes[y_train[i]])

# loading test data
test = pd.read_csv('test.csv')
xtest = np.zeros((test.shape[0],128,128,3))
for i in range(test.shape[0]):
    image = Image.open('Test'+'/'+test['ID'][i])
    image = image.resize((128,128))
    image = np.array(image)
    xtest[i] = image
    
#normalizing data
xtest = (xtest - np.mean(xtrain, axis=0)) / (1e-6 + np.std(xtrain, axis=0))
xtrain = (xtrain - np.mean(xtrain, axis=0)) / (1e-6 + np.std(xtrain, axis=0))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

num_examples = xtrain.shape[0]
train_limit = int(0.8*num_examples)
xval = xtrain[train_limit:]
xtrain = xtrain[:train_limit]
yval = ytrain[train_limit:]
yval = np.array(yval)

print num_examples
print yval.shape
print xval.shape

class Age(nn.Module):
    def __init__(self):
        super(Age,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,4,3,1,1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Conv2d(4,8,3,1,1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Conv2d(8,16,3,1,1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Conv2d(16,32,3,1,1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Conv2d(32,64,3,1,1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            nn.ReLU(),
            
        )
        self.linear = nn.Linear(64*4*4,300)
        self.linear2 = nn.Linear(300,300)
        self.linear3 = nn.Linear(300,3)
        
    def forward(self,input):
        output = self.cnn(input)
        output = output.view(-1,64*4*4)
        output = F.relu(self.linear(output))
        output = F.relu(self.linear2(output))
        return self.linear3(output)
    
    
model = Age()
model.cuda()
lr = 0.001
optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=0.5)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 50
batch_size = 64


for epoch in range(num_epochs):
    error = 0
    for i in range(0,train_limit,batch_size):
        last_index = min(train_limit,i+batch_size)
        batch = xtrain[i:last_index]
        ybatch = ytrain[i:last_index]
        ybatch=np.array(ybatch)
        optimizer.zero_grad()
        out = model(Variable(torch.from_numpy(np.array(batch).transpose(0,3,1,2)).type(torch.FloatTensor).cuda()))
        loss = loss_fn(out,Variable(torch.from_numpy(ybatch).cuda()))
        loss.backward()
        error += loss.data[0]
        optimizer.step()
    print epoch,error
    if epoch%5==0:
        lr = lr * 0.8
        optimizer = optim.Adam(model.parameters(),lr=lr)
    modelname = 'age'+str(epoch)+'.pth'
    torch.save(model.state_dict(), modelname)
    val_acc = 0
    val_label = np.zeros(len(xval))
    for i in range(0,len(xval),batch_size):
        last_index = min(i+batch_size,len(xval))
        batch = xval[i:last_index]
        out = model(Variable(torch.from_numpy(np.array(batch).transpose(0,3,1,2)).type(torch.FloatTensor).cuda()))
        out = out.data.cpu().numpy()
        target = np.argmax(out,1)
        val_label[i:last_index] = target
    val_acc = sum(val_label==yval)/float(len(xval))
    print "Accuracy :",val_acc



model = Age()
model.load_state_dict(torch.load('age47.pth'))
model.cuda()
reverse_classes = {0:'YOUNG', 1:'MIDDLE', 2:'OLD'}
result = np.zeros(test.shape[0])
for i in range(0,test.shape[0],batch_size):
    last_index = min(test.shape[0],i+batch_size)
    batch = xtest[i:last_index]
    out = model(Variable(torch.from_numpy(np.array(batch).transpose(0,3,1,2)).type(torch.FloatTensor).cuda()))
    out = out.data.cpu().numpy()
    target = np.argmax(out,1)
    result[i:last_index] = target
    
ytest = []
for i in range(test.shape[0]):
    ytest.append(reverse_classes[result[i]])
    
mydata = [['Class','ID']]
for i in range(test.shape[0]):
    mydata.append([ytest[i],test['ID'][i]])
    
    import csv
with open('submission.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(mydata)