from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# mandipriceseries = mandipriceseries['2013-02-01':'2013-10-24']
# retailpriceseries = retailpriceseries['2013-02-01':'2013-10-24']

from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries
from averagemandi import expectedarrivalseries
from averagemandi import expectedmandiprice
from averageretail import retailpriceseries

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)


mp = mandipriceseries
rp = retailpriceseries
idx = mp.index
# mp = mandipriceseries.resample('W').mean()
# rp = retailpriceseries.resample('W').mean()

idx,mp,rp = shuffle(idx,mp,rp)

# X = np.array(mp)
# Y = np.array(rp)
# Y = rp.groupby([mp.round(-2)]).min()
# s# X = np.sqrt(X)
# Y = np.sqrt(Y)
# df = pd.DataFrame(columns = ['a','b'])
# df['a'] = mandipriceseries
# df['b'] = np.square(mandipriceseries)

train_size = (int)(0.80 * len(mp))
train = mp[:train_size]
train_labels = rp[:train_size]
test = mp[train_size:]
test_labels = rp[train_size:]


train = train_labels.groupby([train.round(-1)]).min()
test = test_labels.groupby([test.round(-1)]).min()


#regr = linear_model.Ridge(alpha = 0.5)
regr = linear_model.LinearRegression()
regr.fit(np.array(train.index).reshape(-1,1), train)

# predicted_labels = regr.predict(np.array(test.index).reshape(-1,1))
# print('Variance score: %.2f' % r2_score(test, predicted_labels ))
# print("Mean squared error: %.2f" % mean_squared_error(test, predicted_labels))
# plt.scatter(test.index, test,  color='black')
# plt.plot(test.index, predicted_labels, color='red', linewidth=3)


predicted_labels = regr.predict(np.array(train.index).reshape(-1,1))
print('Variance score: %.2f' % r2_score(train, predicted_labels ))
print("Mean squared error: %.2f" % mean_squared_error(train, predicted_labels))
plt.scatter(train.index, train,  color='black')
plt.plot(train.index, predicted_labels, color='red', linewidth=3)

# predicted_labels = regr.predict(test)
# print('Regression Cefficient', regr.coef_ , regr.intercept_)
# print("Mean squared error: %.2f" % mean_squared_error(test_labels, predicted_labels))

# plt.scatter(test['a'], test_labels,  color='black')
# plt.plot(test['a'], predicted_labels, color='blue', linewidth=3)
plt.title('Train Set Results $R^2 = $'+str(0.85))
plt.xlabel('Mandi Price')
plt.ylabel('Retail Price')
#plt.text(3,8, r'$R^2 = $'+str(0.85), fontsize=15)

# plt.xticks(())
# plt.yticks(())

plt.show()