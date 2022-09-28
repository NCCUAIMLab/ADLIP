import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import random
import numpy as np
from torchvision import models
import torch.nn.functional as F

EPOCH = 50
BATCH_SIZE = 50           
LR =0.001                 
DOWNLOAD_MNIST = False    
if_use_gpu = 1            

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  
        self.conv1 = nn.Sequential(
            nn.Conv2d(           
                in_channels=1,  
                out_channels=16, 
                kernel_size=5,   
                stride=1,        
                padding=2,                          
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(), 
            nn.MaxPool2d(2)  
        )
        self.out = nn.Linear(32*7*7, 10) 
       
    def forward(self,x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = x.view(x.size(0), -1)
       output = self.out(x)
       return output

class VGGNet(nn.Module):
    def __init__(self, num_classes=10):	   
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)   
        net.classifier = nn.Sequential()	
        self.features = net		
        self.classifier = nn.Sequential(    
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(), 
    download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

img_size = 28*28
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    
    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    if if_use_gpu:
        cnn = cnn.cuda()
    
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x, requires_grad=False)
            b_y = Variable(y, requires_grad=False)
            if if_use_gpu:
                b_x = b_x.cuda()
                b_y = b_y.cuda()      
            output = cnn(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
            if step % 1000 == 0:
                print('Epoch:', epoch, '|step:', step, '|train loss:%.4f'%loss.data) #每100steps輸出一次train loss

    torch.save(cnn.state_dict(), "./cnn.pt")
    
    device = torch.device("cuda")
    cnn.load_state_dict(torch.load("./cnn.pt", map_location="cuda:0"))  # Choose whatever GPU device number you want
    cnn.to(device)
    
    error = 0

    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        output = cnn(b_x)

        result = torch.argmax(output,dim=1)
        
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")

    error = 0
    # 10% noisy
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        for i in range(50):
            ran_seq = random.sample([n for n in range(img_size)], np.int32(img_size*0.1))
            x = b_x[i].reshape(-1,img_size)
            x[0][ran_seq]=1
            c = torch.reshape(x,(28,28))
            b_x[i] = c
        output = cnn(b_x)
        result = torch.argmax(output,dim=1)
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("Noisy 10%")
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")


    error = 0
    # 20% noisy
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        for i in range(50):
            ran_seq = random.sample([n for n in range(img_size)], np.int32(img_size*0.2))
            x = b_x[i].reshape(-1,img_size)
            x[0][ran_seq]=1
            c = torch.reshape(x,(28,28))
            b_x[i] = c
        output = cnn(b_x)
        result = torch.argmax(output,dim=1)
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("Noisy 20%")
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")

    error = 0
    # 30% noisy
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        for i in range(50):
            ran_seq = random.sample([n for n in range(img_size)], np.int32(img_size*0.3))
            x = b_x[i].reshape(-1,img_size)
            x[0][ran_seq]=1
            c = torch.reshape(x,(28,28))
            b_x[i] = c
        output = cnn(b_x)
        result = torch.argmax(output,dim=1)
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("Noisy 30%")
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")


    vgg = VGGNet()
    optimizer = torch.optim.Adam(vgg.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    if if_use_gpu:
        vgg = vgg.cuda()
    for epoch in range(10):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x, requires_grad=False)
            b_y = Variable(y, requires_grad=False)
            if if_use_gpu:
                b_y = b_y.cuda()      
            b_c = torch.zeros([50,3,28,28])
            for i in range(len(b_x)):
                c = torch.cat((b_x[i],b_x[i],b_x[i]),0)
                b_c[i] = c
            b_c = F.interpolate(b_c,scale_factor=2,mode="bilinear", align_corners=True)
            b_c = b_c.cuda()
            output = vgg(b_c)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
            if step % 1000 == 0:
                print('Epoch:', epoch, '|step:', step, '|train loss:%.4f'%loss.data) #每100steps輸出一次train loss

    torch.save(vgg.state_dict(), "./vgg.pt")
    
    device = torch.device("cuda")
    vgg.load_state_dict(torch.load("./vgg.pt", map_location="cuda:0"))  # Choose whatever GPU device number you want
    vgg.to(device)
    error = 0
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        b_c = torch.zeros([50,3,28,28])
        for i in range(len(b_x)):
            c = torch.cat((b_x[i],b_x[i],b_x[i]),0)
            b_c[i] = c
        b_c = b_c.cuda()
        b_c = F.interpolate(b_c,scale_factor=2,mode="bilinear", align_corners=True)
        output = vgg(b_c)

        result = torch.argmax(output,dim=1)
        
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")


    error = 0
    # 10% noisy
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        
        b_c = torch.zeros([50,3,28,28])

        for i in range(50):
            ran_seq = random.sample([n for n in range(img_size)], np.int32(img_size*0.1))
            x = b_x[i].reshape(-1,img_size)
            x[0][ran_seq]=1
            c = torch.reshape(x,(28,28))
            b_x[i] = c
            c = torch.cat((b_x[i],b_x[i],b_x[i]),0)
            b_c[i] = c

        b_c = b_c.cuda()
        b_c = F.interpolate(b_c,scale_factor=2,mode="bilinear", align_corners=True)
        output = vgg(b_c)
        result = torch.argmax(output,dim=1)
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("Noisy 10%")
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")


    error = 0
    # 20% noisy
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        
        b_c = torch.zeros([50,3,28,28])

        for i in range(50):
            ran_seq = random.sample([n for n in range(img_size)], np.int32(img_size*0.2))
            x = b_x[i].reshape(-1,img_size)
            x[0][ran_seq]=1
            c = torch.reshape(x,(28,28))
            b_x[i] = c
            c = torch.cat((b_x[i],b_x[i],b_x[i]),0)
            b_c[i] = c
        b_c = b_c.cuda()
        b_c = F.interpolate(b_c,scale_factor=2,mode="bilinear", align_corners=True)
        output = vgg(b_c)
        result = torch.argmax(output,dim=1)
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("Noisy 20%")
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")



    error = 0
    # 30% noisy
    for step, (x, y) in enumerate(test_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()      
        
        b_c = torch.zeros([50,3,28,28])

        for i in range(50):
            ran_seq = random.sample([n for n in range(img_size)], np.int32(img_size*0.3))
            x = b_x[i].reshape(-1,img_size)
            x[0][ran_seq]=1
            c = torch.reshape(x,(28,28))
            b_x[i] = c
            c = torch.cat((b_x[i],b_x[i],b_x[i]),0)
            b_c[i] = c
        b_c = b_c.cuda()
        b_c = F.interpolate(b_c,scale_factor=2,mode="bilinear", align_corners=True)
        output = vgg(b_c)
        result = torch.argmax(output,dim=1)
        A = result.tolist()
        B = b_y.tolist()
        for i in range(len(A)):
            if A[i] != B[i]:
                error+=1
    
    error_rate = error/10000
    print("Noisy 30%")
    print("The error rate is ", error_rate*100,"%")
    print("The accuracy rate is ", (1-error_rate)*100,"%")
