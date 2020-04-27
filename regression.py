import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

boston=load_boston()
print(boston.DESCR)

dataset=boston.data
for name,index in enumerate(boston.feature_names):
  print(index,name)
  
data=dataset[:,12].reshape(-1,1)
np.shape(dataset)
target=boston.target.reshape(-1,1)
np.shape(target)

%matplotlib inline
plt.scatter(data,target,color='green')
plt.xlabel('Lower income population')
plt.ylabel('cost of House')
plt.show()

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(data,target)
pred=reg.predict(data)

%matplotlib inline
plt.scatter(data,target,color='red')
plt.plot(data,pred,color='green')
plt.xlabel('Lower income population')
plt.ylabel('cost of House')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model=make_pipeline(PolynomialFeatures(3),reg)
model.fit(data,target)
pred=model.predict(data)

from sklearn.metrics import r2_score
r2_score(pred,target)
