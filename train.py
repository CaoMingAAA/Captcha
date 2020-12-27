import torch
import torch.nn as nn
from torch.autograd import Variable
from model import CNN
from data import CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import time
import os
import pandas as pd
from sklearn.utils import shuffle as reset
import matplotlib.pyplot as plt

eps = 1e-5
batch_size = 128
base_lr = 0.001
max_epoch = 50

MODEL_PATH = r"./checkpoints/model.pth"
TRAIN_PATH = r"./data/train/"

restor = False

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


def train_test_split(df, test_size=0.3, shuffle=False, random_state=None):
    if shuffle:
        df = reset(df, random_state=random_state)
    traindf = df[int(len(df) * test_size):]
    testdf = df[:int(len(df) * test_size)]

    return traindf, testdf

def calculat_acc(output, target):
    output, target = output.view(-1, 62), target.view(-1, 62)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 4), target.view(-1, 4)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = float(sum(correct_list) / len(correct_list))
    return acc


def cal_acc_loss(acc_history, loss_history):
    loss = torch.mean(torch.Tensor(loss_history))
    acc = torch.mean(torch.Tensor(acc_history))
    return float(acc), float(loss)

def load_data():
    """
    :return: DataLoader
    """
    # merge operate, convert a image from PIL to tensor.
    transforms = Compose([ToTensor()])
    df = pd.read_csv('./data/label.csv', index_col='ID')

    # split dataSet
    df_train, df_test = train_test_split(df)
    # generate train data.
    train_dataset = CaptchaData(TRAIN_PATH, transform=transforms, df=df_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                   shuffle=True, drop_last=True)
    # generate test data.
    test_dataset = CaptchaData(TRAIN_PATH, transform=transforms, df=df_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                  num_workers=0, shuffle=True, drop_last=True)
    return train_data_loader, test_data_loader


def train():

    train_data_loader, test_data_loader = load_data()
    cnn = CNN()
    if torch.cuda.is_available():
        cnn.cuda()
    if restor:
        cnn.load_state_dict(torch.load(MODEL_PATH))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=base_lr)  # optimize
    criterion = nn.MultiLabelSoftMarginLoss()

    train_acc_epoch = []
    test_acc_epoch = []
    loss_epoch = []

    for epoch in range(max_epoch):
        start = time.time()
        cnn.train()
        acc_history = []
        loss_history = []
        # in train set
        for img, target in train_data_loader:
            img = Variable(img)
            target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_history.append(calculat_acc(output, target))
            loss_history.append(float(loss))

        train_acc, loss = cal_acc_loss(acc_history, loss_history)

        loss_history = []
        acc_history = []
        cnn.eval()

        # in test set
        for img, target in test_data_loader:
            img = Variable(img)
            target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            acc_history.append(calculat_acc(output, target))

        test_acc, nothing = cal_acc_loss(acc_history, loss_history)

        print('train loss:{:.5f}'.format(loss))
        print('train acc:{:.5f}'.format(train_acc))
        print('test acc: {:.5f}'.format(test_acc))
        print('epoch: {} , using time: {:.5}\n'.format(epoch + 1, time.time() - start))

        train_acc_epoch.append(train_acc)
        test_acc_epoch.append(test_acc)
        loss_epoch.append(loss)
        torch.save(cnn.state_dict(), MODEL_PATH)
        if loss < eps:
            break

    return train_acc_epoch, test_acc_epoch, loss_epoch


trainacc, testacc, loss = train()
plt.title('Accuracy')
plt.plot(trainacc, label='Train accuracy')
plt.plot(testacc, label='Test accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()
plt.show()

plt.title('train loss')
plt.plot(loss, label='Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.show()

print("final train accuracy after {:} times: {:.5f}".format(len(trainacc), trainacc[-1]))
print("final test accuracy after {:} times: {:.5f}".format(len(testacc), testacc[-1]))
print("final loss after {:} times: {:.5f}".format(len(loss), loss[-1]))


pass

