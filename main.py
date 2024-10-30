import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, tree, linear_model


df = pd.read_excel("ssda.xlsx", sheet_name='Лист1', header=0)
vegetables = ['Не опр.', 'Медленная скор.', 'Хор.']
counts1 = 0
counts2 = 0
counts3 = 0
for i in df.Дефект:
    if i == 'Не определяетя':
        counts1 += 1
    elif i == 'Медленная скорость чтения записи':
        counts2 += 1
    elif i == 'Хороший':
        counts3 += 1
counts = [counts1, counts2, counts3]
plt.bar(vegetables, counts)
plt.title('Дефекты дисков')
plt.xlabel('Дефект')
plt.ylabel('Количество')
plt.show()



df = pd.read_excel("ssda.xlsx", sheet_name='Лист1', header=0)
countsgb1 = countsgb2 = countsgb3 = countsgb4 = countsgb5 = countsgb6 = 0
for i in df.Размер:
    if i == 2000:
        countsgb1 += 1
    elif i == 1000:
        countsgb2 += 1
    elif i == 480:
        countsgb3 += 1
    elif i == 500:
        countsgb4 += 1
    elif i == 256:
        countsgb5 += 1
    elif i == 128:
        countsgb6 += 1

plt.title('Разделение дисков по Размеру ГБ')
plt.pie([countsgb1, countsgb2, countsgb3, countsgb4, countsgb5, countsgb6], labels=['2000Gb', '1000Gb', '480Gb', '500Gb', '256Gb', '128Gb']) # круговая диаграмма
plt.show()




df = pd.read_excel("ssda.xlsx", sheet_name='Лист1', header=0)
plt.scatter(df.Размер, df.Цена, color='red', marker='+')
plt.xlabel('Размер, гб')
plt.ylabel('Цена, р.')
plt.title('Значение цены в зависимости от параметра размер')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['Размер']], df.Цена)
print(reg.predict(pd.DataFrame({'Размер':[1000]})))

plt.scatter(df.Размер, df.Цена, color='red', marker='+')
plt.xlabel('Размер, гб')
plt.ylabel('Цена, р.')
plt.plot(df.Размер, reg.predict(df[['Размер']]))
plt.title('Среднее значение цены в зависимости от параметра размер')
plt.show()

pred = pd.read_excel('ssda.xlsx')

p = (reg.predict(pred))


pred['price'] = p

print(pred)


cols = df.columns[:4] # выбираем первые 4 колонки
colours = ['#000099', '#ffff00'] # определите цвета - желтые – это пропущенные. синие - не пропущенные.
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
