import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, input_dim = 400, hidden_dims = [120,84], output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.classifier = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))      
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        y = self.classifier(x)
        return y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 64
name = "avg_pool"


transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10('data/cifar10/', train=True, transform=transform_all)
test_dataset = datasets.CIFAR10('data/cifar10/', train=False, transform=transform_all)

train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


model = SimpleCNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,weight_decay=1e-5,momentum=0.9,nesterov=True)
model_h = SimpleCNN().to(device)

from pyhessian.hessian import hessian

criterion = torch.nn.CrossEntropyLoss().to(device)

def train():

    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        

        loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()

        

    acc = test()
    temp_loss = loss.cpu().detach().numpy()
   

    model_h.load_state_dict(model.state_dict())
    model_h.eval()
    hessian_comp = hessian(model_h, criterion, dataloader=train_loader, device=device)
    trace,_ = hessian_comp.trace()


        
    # return loss_list,acc_list_test
    return temp_loss,acc,np.mean(trace) 
        
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    #print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


EPOCH = 50
if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn
    


    loss_list = []
    acc_list_test = []
    trace_list = []
    for epoch in range(EPOCH):
        loss,acc,trace = train()
        print(epoch,loss,acc,trace)
        loss_list.append(loss)
        acc_list_test.append(acc)
        trace_list.append(trace)


    torch.save(model.state_dict(), f'{name}.pt')

    import csv
    rows = zip(loss_list,acc_list_test,trace_list)
    
    with open(f"{name}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for row in rows:
                writer.writerow(row)    