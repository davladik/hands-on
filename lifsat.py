import matplotlib.pyplot as plt
# import mplcyberpunk as mpl
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


data = 'data\lifesat\lifesat.csv'
lifesat = pd.read_csv(data)
X = lifesat[['GDP per capita (USD)']].values
y = lifesat[['Life satisfaction']].values

lifesat.plot(kind='scatter', grid=True, style='cyberpunk',
             x='GDP per capita (USD)', y='Life satisfaction')

plt.axis([23_500, 62_500, 4, 9])


model = LinearRegression()
model.fit(X, y)
X_new = [[37_655.2]]


lin_func = lambda x: 3.74904943 + 6.77889969e-05 * x
x = np.linspace(25000, 62000, 1000)
y1 = [lin_func(i) for i in x]

# plt.style.use('cyberpunk')
plt.plot(x, y1)
# mpl.make_lines_glow()
# mpl.make_scatter_glow()
plt.show()


print(model.predict(X_new))