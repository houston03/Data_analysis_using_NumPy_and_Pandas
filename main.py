import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
import sklearn
from sklearn import svm, tree, linear_model

# распределение в один из классов
df = pd.read_excel("ner.xlsx")
print(df)

# %matplotlib inline
plt.scatter(df.score, df.price, color='red', marker='+')
plt.xlabel('score, см')
plt.ylabel('price, см')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['score']], df.price)
print(reg.predict(pd.DataFrame({'score':[9]})))

plt.scatter(df.score, df.price, color='red', marker='+')
plt.xlabel('score, см')
plt.ylabel('price, см')
plt.plot(df.score, reg.predict(df[['score']]))
plt.show()

pred = pd.read_excel('sco.xlsx')

p = (reg.predict(pred))


pred['price'] = p
# среднее значение цены в зависимости от параметра score
print(pred)


'''
df = pd.read_excel("test.xlsx")
print(df)

columns_target = ['y/n']

columns_train = ['price', 'sost', 'vlgn', 'sum']

x = df[columns_train]
y = df[columns_target]

# проверка на пустые строчки
# print(x['price'].isnull().sum())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier(max_depth=5, random_state=21)


bagging = BaggingClassifier(estimator=clf, n_estimators=100)
bagging.fit(x_train, y_train)
# проверка точности
print(bagging.score(x_test, y_test))


clf.fit(x_train, y_train)
# проверка точности
print(clf.score(x_test, y_test))



from sklearn import svm, tree

predmodel = svm.LinearSVC()
predmodel.fit(x_train, y_train)


print(predmodel.predict(x_test[0:7]))
# проверка точности
print(predmodel.score(x_test, y_test))


def act(x):
    return 0 if x < 0.5 else 1

def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])  # матрица 2x3
    weight2 = np.array([-1, 1])     # вектор 1х2

    sum_hidden = np.dot(weight1, x)       # вычисляем сумму на входах нейронов скрытого слоя
    print("Значения сумм на нейронах скрытого слоя: "+str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: "+str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Выходное значение НС: "+str(y))

    return y

house = 1
rock = 1
attr = 1

res = go(house, rock, attr)
if res == 1:
    print("Ты мне нравишься")
else:
    print("Созвонимся")

'''