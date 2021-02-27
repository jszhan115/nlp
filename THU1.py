c# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:37:22 2021

@author: Jszhan
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:05:31 2021

@author: Jszhan
加上dropout，和relu效果并不好
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


train_data=[]
train_label=[]
with open('data/shuffle_Train_IDs.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                l1=[]
                s=data.split(',')
                for n in range(len(s)):
                    l1.append(int(s[n]))
                train_data.append(l1)
                train_label.append([int(label)])
#print(train_data[1000])              
#print(train_label[1000])

val_data=[]
val_label=[]
with open('data/Val_IDs.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                data, label = line.split('\t')
                l1=[]
                s=data.split(',')
                for n in range(len(s)):
                    l1.append(int(s[n]))
                val_data.append(l1)
                val_label.append([int(label)])
#print(val_data[1000])              
#print(val_label[1000])

hidden_size=256
dict_size=5307
device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
accuracy=0.
path='./model.pth'

class RNN(nn.Module):
    def __init__(self, dict_size, hidden_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size

        
        self.embedding=nn.Embedding(dict_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        #self.fc1=nn.Linear(hidden_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,14)
        

    def forward(self, input, hidden):
        embed=self.embedding(input).unsqueeze(1)
        output=embed
        output,hidden=self.gru(output,hidden)
        #output=F.dropout(output,0.5)
        #output=F.relu(self.fc1(output[-1]))
        #output=F.dropout(output,0.5)
        output=self.fc2(output[-1])
        output=F.log_softmax(output,dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1, self.hidden_size,device=device)

rnn=RNN(dict_size,hidden_size).to(device)

criterion=F.nll_loss
learning_rate=0.01
optimizer=optim.SGD(rnn.parameters(),lr=learning_rate)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    output, hidden = rnn(line_tensor, hidden)
    #output=torch.unsqueeze(output,0)
    #category_tensor=torch.unsqueeze(category_tensor, 0)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

def evaluate_tensor(eval_tensor,eval_lab):
    hidden=rnn.initHidden()
    output,_=rnn(eval_tensor,hidden)
    result=torch.argmax(output,dim=1)
    loss=criterion(output, eval_lab)
    return result.item()==eval_lab.item(),loss.item()

def evaluate_all(eval_data,eval_label):
    with torch.no_grad():
        correct=0.
        all_loss=0.
        for i in range(len(eval_label)):
            label1=torch.tensor(eval_label[i]).to(device)
            data1=torch.tensor(eval_data[i]).to(device)
            c,loss=evaluate_tensor(data1, label1)
            correct+=c
            all_loss+=loss
        acc=correct/len(eval_label)
        global accuracy
        if acc>accuracy:
            accuracy=acc
            torch.save(rnn.state_dict(),path)
        avg_loss=all_loss/len(eval_label)
        print("val acc:{},val loss:{:10.6f}".format(acc,avg_loss))
        print()

n_iters=100000
print_every=5000

current_loss=0.
for iter in range(1, n_iters + 1):
    label=torch.tensor(train_label[iter%len(train_label)]).to(device)
    data=torch.tensor(train_data[iter%len(train_data)]).to(device)
    output, loss = train(label, data)
    current_loss+=loss
    

    # Print iter number, loss
    if iter==1:
        print('iter:{},loss:{:10.6f}'.format(iter,loss))
        evaluate_all(val_data,val_label)
    if iter % print_every == 0:
        print('iter:{},loss:{:10.6f}'.format(iter,current_loss/print_every))
        current_loss=0.
        evaluate_all(val_data,val_label)

print("acurracy:{:10.6f}".format(accuracy))        
rnn.load_state_dict(torch.load(path))
test_data=[]
with open('data/Test_IDs.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                l1=[]
                s=line.split(',')
                for n in range(len(s)):
                    l1.append(int(s[n]))
                test_data.append(l1)
                
#print(test_data[1000])              

def predict_tensor(test_tensor):
    hidden=rnn.initHidden()
    output,_=rnn(test_tensor,hidden)
    result=torch.argmax(output,dim=1)
    return result.item()

results=[]
def predict_all(test_data):
    with torch.no_grad():
        for i in range(len(test_data)):
            test_tensor=torch.tensor(test_data[i]).to(device)
            result=predict_tensor(test_tensor)
            results.append(result)

predict_all(test_data)
with open('data/results.txt','w') as f:
    f.seek(0)
    f.truncate()
with open('data/results.txt','a') as f:
    s=''
    for i in range(len(results)):
        s=s+str(results[i])+','
        if (i+1)%100 == 0:
            s+='\n'
            f.write(s)
            s=''
    f.write(s+"finished")
        
        
            
