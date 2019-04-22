import numpy as np
import csv
import matplotlib.pyplot as plt

# original import data
# data = []
# for i in range(18):
#     data.append([])
# with open("data/train.csv") as file:
#     reader = csv.reader(file, delimiter=',')
#     next(reader)
#     reader = list(reader)
#     for i in range(len(reader)):
#         row = reader[i][3:]
#         for j in range(24):
#             if row[j] == 'NR':
#                 data[i % 18].append(float(0))
#             else:
#                 data[i % 18].append(float(row[j]))

# original parse data
# x_data = []
# y_data = []
# for day in range(240):
#     for i in range(15):
#         # x_data.append(data[9][15 * day + i:15 * day + 9 + i])
#         # only pm2.5 not good as add 18 types
#         x_data.append([])
#         for t in range(18):
#             x_data[15 * day + i] += data[t][15 * day + i:15 * day + 9 + i]
#             # add **2 will decrease loss more, will it overfit?
#             # x_data[16 * day + i] += [val**2 for val in data[t][16 * day + i:16 * day + 8 + i]]
#         y_data.append(data[9][15 * day + 9 + i])
# x_data = np.array(x_data)
# y_data = np.array(y_data)
# print(x_data.shape)
# # add bias 1 to each x_data
# bias = np.ones(shape=(len(x_data), 1))
# x_data = np.append(bias, x_data, axis=1)

# improved parsing
raw_data = np.genfromtxt('data/train.csv', delimiter=',')
data = raw_data[1:, 3:]
NaNs = np.isnan(data)
data[NaNs] = 0

data_monthly = {}
for month in range(12):
    dim = np.empty((18, 480))
    for day in range(20):
        for hr in range(24):
            dim[:, day * 24 + hr] = data[18 * (month * 20 + day):18 * (month * 20 + day + 1), hr]
    data_monthly[month] = dim

x_data = np.empty(shape=(12 * 480, 18 * 8 * 2), dtype=float)
y_data = np.empty(shape=(12 * 480, 1), dtype=float)
for month in range(12):
    for day in range(20):
        for hr in range(24):
            if day == 19 and hr > 15:
                continue
            x_data[month * 480 + day * 24 + hr, :] = \
                np.concatenate((data_monthly[month][:, day * 24 + hr:day * 24 + hr + 8].reshape(1, -1),
                                np.power(data_monthly[month][:, day * 24 + hr:day * 24 + hr + 8].reshape(1, -1), 2),
                                # np.power(data_monthly[month][:, day * 24 + hr:day * 24 + hr + 8].reshape(1, -1), 3),
                                ), axis=1)
            y_data[month * 480 + day * 24 + hr, 0] = data_monthly[month][9, day * 24 + hr + 8]

# normalization
mean = np.mean(x_data, axis=0)
std = np.std(x_data, axis=0)
for i in range(x_data.shape[0]):
    for j in range(x_data.shape[1]):
        if not std[j] == 0:
            x_data[i][j] = (x_data[i][j] - mean[j]) / std[j]

x_data = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1).astype(float)


def gradient_descent(x, y, lr, iteration):
    n = x.shape[0] - 1
    w = np.zeros((x.shape[1], 1))
    cost_list = []
    iterations_list = []
    for k in range(iteration):
        cost = np.sum(np.power(x.dot(w) - y, 2)) / n
        gradient = (-2) * np.transpose(x).dot(y - x.dot(w))
        w = w - lr * gradient
        cost_list.append(cost)
        iterations_list.append(k)
    print(cost_list[-1])
    plt.plot(iterations_list[3:], cost_list[3:], color='red')


def adagrad(x, y, lr, iteration):
    result = []
    w = np.zeros((x.shape[1], 1))
    cost_list = []
    iterations_list = []
    adagrad_sum = np.zeros((x.shape[1], 1))
    learning_rate = np.array([[lr]] * x.shape[1])
    for k in range(iteration):
        y_predicted = np.dot(x, w)
        cost = np.sum(np.power(x.dot(w) - y, 2)) / (x.shape[0] - 1)
        gradient = (-2) * np.transpose(x).dot(y - x.dot(w))
        adagrad_sum += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad_sum)
        cost_list.append(cost)
        iterations_list.append(k)
        result.append(y_predicted)
    print(cost_list[-1])
    np.save('data/weight.npy', w)
    # plt.plot(iterations_list[3:], cost_list[3:], color='green')


def stochastic_gradient_descent(x, y, lr, iteration):
    n = x.shape[0] - 1
    w = np.zeros((x.shape[1], 1))
    cost_list = []
    iterations_list = []
    for k in range(iteration):
        random = np.random.randint(0, n)
        y_predicted = x.dot(w)
        cost = np.sum(np.power(y_predicted - y, 2)) / n
        y_sample = np.dot(x[random], w)
        gradient = -2 * np.dot(x[random].T.reshape(-1, 1), y[random] - y_sample)
        w = w - (lr * gradient).reshape(-1, 1)
        cost_list.append(cost)
        iterations_list.append(k)
    print(cost_list[-1])
    plt.plot(iterations_list[3:], cost_list[3:], color='blue')


# training
# gradient_descent(x_data, y_data, lr=0.000001, iteration=10000)
adagrad(x_data, y_data, lr=1, iteration=10000)
# stochastic_gradient_descent(x_data, y_data, lr=0.001, iteration=10000)

# read test
weight = np.load('data/weight.npy')
test = np.genfromtxt('data/test.csv', delimiter=',')[:, 2:-1]
nans = np.isnan(test)
test[nans] = 0
test_x = np.empty((240, 18 * 8 * 2))
actual = np.empty((240, 1))
for i in range(240):
    test_x[i, :] = \
        np.concatenate((test[18 * i:18 * (i + 1)].reshape(1, -1),
                        np.power(test[18 * i:18 * (i + 1)].reshape(1, -1), 2),
                        # np.power(test[18 * i:18 * (i + 1)].reshape(1, -1), 3),
                        ), axis=1)

answer = np.genfromtxt('data/test.csv', delimiter=',')[:, -1]

for i in range(240):
    actual[i] = answer[18 * i + 9]
for i in range(test_x.shape[0]):
    for j in range(test_x.shape[1]):
        if not std[j] == 0:
            test_x[i][j] = (test_x[i][j] - mean[j]) / std[j]

test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1).astype(float)
answer = test_x.dot(weight)
f = open("data/result", "w")
writer = csv.writer(f)
title = ['id', 'value']
name = []
writer.writerow(title)
for i in range(240):
    writer.writerow(['id_' + str(i), answer[i]])
    name.append(i)
test_cost = np.sum(np.power(answer - actual, 2)) / 240
print(test_cost)

plt.plot(actual, 'r.')
plt.bar(name, np.ndarray.flatten(answer))

plt.show()
