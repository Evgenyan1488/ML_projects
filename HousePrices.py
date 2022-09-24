import torch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
from scipy.special import boxcox1p
from sklearn.model_selection import KFold, cross_val_score, train_test_split

device = 'cuda'
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 20)

raw_data = pd.read_csv("CSV/train.csv")
t_raw_data = pd.read_csv("CSV/test.csv")

# Data processing

raw_data.drop("Id", axis=1, inplace=True)
t_raw_data.drop("Id", axis=1, inplace=True)


# Обработка string данных:
# Заменяем Nan

obj_indexes = raw_data.dtypes[raw_data.dtypes == "object"].index
obj_data = raw_data[obj_indexes]
obj_data = obj_data.apply(lambda x: x.fillna("N/A"), axis=0)


t_obj_indexes = t_raw_data.dtypes[t_raw_data.dtypes == "object"].index
t_obj_data = t_raw_data[t_obj_indexes]
t_obj_data = t_obj_data.apply(lambda x: x.fillna("N/A"), axis=0)
# Заменяем буквы на цифры

# for col in obj_indexes:
#     unique_values = pd.unique(obj_data[col])
#     for index, value in enumerate(unique_values):
#         obj_data[col] = obj_data[col].replace(value, index + 1)

mapping_table = dict()
for col in obj_indexes:

    curr_mapping_table = dict()
    unique_values = pd.unique(obj_data[col])

    for index, value in enumerate(unique_values):
        curr_mapping_table[value] = index + 1
        obj_data[col] = obj_data[col].replace(value, index + 1)
        #print(obj_data[col])

    mapping_table[col] = curr_mapping_table


for col in mapping_table.keys():
    curr_mapping_table = mapping_table[col]
    #print(curr_mapping_table)
    for k, v in curr_mapping_table.items():
        t_obj_data[col] = t_obj_data[col].replace(k, v)

t_obj_data = t_obj_data.replace('N/A', 0)

# Обработка numeric данных

num_indexes = raw_data.dtypes[raw_data.dtypes != "object"].index
num_data = raw_data[num_indexes]


t_num_indexes = t_raw_data.dtypes[t_raw_data.dtypes != "object"].index
t_num_data = t_raw_data[t_num_indexes]

# Заменяем Nan значения на 0, ибо это логично

num_data = num_data.apply(lambda x: x.fillna(0), axis=0)


t_num_data = t_num_data.apply(lambda x: x.fillna(0), axis=0)

# Объединяем numeric + string (obj) и нормализуем

norm_data = pd.concat([obj_data, num_data], axis=1)

t_norm_data = pd.concat([t_obj_data, t_num_data], axis=1)

max_val, mean_val, min_val = dict(), dict(), dict()

for col in norm_data:
    max_val[col] = norm_data[col].max()
    mean_val[col] = norm_data[col].mean()
    min_val[col] = norm_data[col].min()


norm_data = (norm_data - norm_data.mean()) / (norm_data.max() - norm_data.min())
#t_norm_data = (t_norm_data - norm_data.mean()) / (norm_data.max() - norm_data.min())
#print(t_norm_data.describe())

# print(t_norm_data.columns)
# print(norm_data.columns)


for col in t_norm_data.columns:
    t_norm_data[col] = (t_norm_data[col] - mean_val[col]) / (max_val[col] - min_val[col])

numeric_x_columns = list(norm_data.columns)
numeric_x_columns.remove('SalePrice')

t_numeric_x_columns = list(t_norm_data.columns)


# numeric_y_columns = ['SalePrice']

dfx = pd.DataFrame(norm_data, columns=numeric_x_columns)   # Поч тут ремув не бачит???
dfy = pd.DataFrame(norm_data, columns=['SalePrice'])

t_dfx = pd.DataFrame(t_norm_data, columns=t_numeric_x_columns)   # Поч тут ремув не бачит???



x_all_processed = torch.tensor(dfx.values, dtype=torch.float)
y_all_processed = torch.tensor(dfy.values, dtype=torch.float)

x_train, x_test, y_train, y_test = train_test_split(x_all_processed, y_all_processed, train_size=0.66)




t_x_data = torch.tensor(t_dfx.values, dtype=torch.float)
#print(t_norm_data)


# Modeling

def predict(net, x, y):
    y_predict = net.forward(x)

    plt.plot(x, y, 'o', c='g', label='То что надо')
    plt.plot(x, y_predict.data, 'o', c='r', label='То что имеем')
    plt.legend(loc='upper left')


def loss(pred, true):
    sq = (pred-true)**2
    return sq.mean()


class ClampNet(torch.nn.Module):

    def __init__(self, N_In, N1, N2, N3):
        super(ClampNet, self).__init__()
        self.fc1 = torch.nn.Linear(N_In, N1)
        #self.act1 = torch.nn.Sigmoid()

        self.fc2 = torch.nn.Linear(N1, N2)
        #self.act2 = torch.nn.Sigmoid()

        self.fc3 = torch.nn.Linear(N2, N3)
        #self.act3 = torch.nn.Sigmoid()

        self.fc4 = torch.nn.Linear(N3, 1)

    def forward(self, x):
        x = self.fc1(x).clamp(min=0)
        # x = self.act1(x)
        x = self.fc2(x).clamp(min=0)
        # x = self.act2(x)
        x = self.fc3(x).clamp(min=0)
        # x = self.act3(x)
        x = self.fc4(x)

        return x


class SigmoidNet(torch.nn.Module):

    def __init__(self, N_In, N1, N2, N3):
        super(SigmoidNet, self).__init__()
        self.fc1 = torch.nn.Linear(N_In, N1)
        self.act1 = torch.nn.Sigmoid()

        self.fc2 = torch.nn.Linear(N1, N2)
        self.act2 = torch.nn.Sigmoid()

        self.fc3 = torch.nn.Linear(N2, N3)
        self.act3 = torch.nn.Sigmoid()

        self.fc4 = torch.nn.Linear(N3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)

        return x


our_net = ClampNet(79, 500, 1000, 200).to(device)

criterion = torch.nn.MSELoss(reduction='sum')


optimizer = torch.optim.Adam(our_net.parameters(), lr=1e-4 * 1)

#print(x_learn_data.shape, y_learn_data.shape)

train_loss_arr = []
test_loss_arr = []

for e in range(5000):
    optimizer.zero_grad()
    y_pred = our_net.forward(x_train.to(device).float())
    y_pred_test = our_net.forward(x_test.to(device).float())
    #t_y_pred = our_net.forward(t_x_data.to(device).float())
    train_loss_val = criterion(y_pred, y_train.to(device))
    test_loss_val = criterion(y_pred_test, y_test.to(device))
    train_loss_arr.append(train_loss_val.item())
    test_loss_arr.append(test_loss_val.item())
    if not e % 500:
        #print(y_pred)
        print('train:', train_loss_val.item())
        print('test:', test_loss_val.item())
    train_loss_val.backward()
    optimizer.step()


torch.save(our_net.state_dict(), 'Model/model.pt')

loaded_net = ClampNet(79, 500, 1000, 200)
loaded_net.load_state_dict(torch.load('Model/model.pt'))

test_output = loaded_net(torch.tensor(t_dfx.values, dtype=torch.float)) #t_x_data
test_output = test_output.detach().numpy()

result = pd.DataFrame(test_output, columns=['SalePrice'])

result['SalePrice'] = result['SalePrice'] * (max_val['SalePrice'] - min_val['SalePrice']) + mean_val['SalePrice']

result['Id'] = pd.array(result.index)
result['Id'] = result['Id'] + 1461
result.to_csv('./output.csv', columns=['Id', 'SalePrice'], index=False)

plt.plot(train_loss_arr, label='train')
plt.plot(test_loss_arr, label='test')
plt.legend(loc='upper right')
plt.show()
#raw_data.drop(columns=["SalePrice"], inplace=True)

# Data processing


#   Нахуя-то сохраняли в словарь, но я сделал без этого
# mapping_table = dict()
# for col in obj_indexes:
#
#     curr_mapping_table = dict()
#     unique_values = pd.unique(obj_data[col])
#
#     for index, value in enumerate(unique_values):
#         curr_mapping_table[value] = index + 1
#         obj_data[col] = obj_data[col].replace(value, index + 1)
#
#     print(curr_mapping_table)

#na_num_indexes = num_data.columns[num_data.isnull().any(0)]        Если нужны нулёвые индексы
#na_num_data = num_data[na_num_indexes]                             Или нулёвая дата


#test_data.drop(columns=["SalePrice"], inplace=True)


# OLD VERSION

# numerical_indexes = train_data.dtypes[train_data.dtypes != "object"].index
#
# #print(numerical_indexes)
#
# skew_feats = train_data[numerical_indexes].apply(lambda x: skew(x)).sort_values(ascending=False)
#
# skewness = pd.DataFrame({'Skew': skew_feats})
#
# skewness = skewness[abs(skewness) > 0.75]
# #print(skewness)
#
# skewed_features = skewness.index
#
# lam = 0.15
# for feat in skewed_features:
#     train_data[feat] = boxcox1p(train_data[feat], lam)
#
#
# sale_price = boxcox1p(sale_price, lam)
#
# train_data = pd.get_dummies(train_data)
# train_data = train_data.apply(lambda x: x.fillna(x.mean()), axis=0)



#train_tensor = torch.tensor(train_data.values)
#price_tensor = torch.tensor(sale_price.values)