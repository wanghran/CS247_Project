import time
from utils import accuracy, hybrid_loss

def train(data, model, optimizer, epoch):
    features, adjs, idx_train, labels, label_mapping = data
    for i in range(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output, encoding = model(features, adjs)
        loss_train = hybrid_loss(output[idx_train], encoding[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
  
        print('Epoch: {:04d}'.format(i+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'time: {:.4f}s'.format(time.time() - t))   