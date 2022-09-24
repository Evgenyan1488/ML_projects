import torch
import matplotlib.pyplot as plt
import matplotlib


device = 'cuda'


def predict(net, x, y):
    y_predict = net.forward(x)

    plt.plot(x, y, 'o', c='g', label='То что надо')
    plt.plot(x, y_predict.data, 'o', c='r', label='То что имеем')
    plt.legend(loc='upper left')


def loss(pred, true):
    sq = (pred-true)**2
    return sq.mean()



class OurNet(torch.nn.Module):  # нахуя классу эта хуйня??

    def __init__(self, n_hid_n):
        super(OurNet, self).__init__()  # Че за супер?
        self.fc1 = torch.nn.Linear(1, n_hid_n)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hid_n, n_hid_n)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hid_n, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)

        return x

our_net = OurNet(50).to(device)


matplotlib.rcParams['figure.figsize'] = (15.0, 7.0)

x_train = torch.rand(1000) * 50 - 25
y_train = torch.cos(x_train)**3
y_train.to(device)

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)


x_val = torch.linspace(-25, 25, 500)
y_val = torch.cos(x_val.data)**3  #Нахуя data??

x_val.unsqueeze_(1)
y_val.unsqueeze_(1)


optimizer = torch.optim.Adam(our_net.parameters(), lr=0.01)


for e in range(10000):
    optimizer.zero_grad()
    y_pred = our_net.forward(x_train.to(device))

    loss_val = loss(y_pred, y_train.to(device))

    if not e % 1000:
        print(loss_val)

    loss_val.backward()
    optimizer.step()



#plt.plot(x_val, y_val, 'o')
#plt.plot(x_train, y_train, 'o')

predict(our_net.cpu(), x_val, y_val)
plt.title('$ y = cos^3(x) $')
plt.show()