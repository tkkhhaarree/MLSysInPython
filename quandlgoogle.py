import quandl, math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

df=quandl.get("WIKI/GOOGL")
df=df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['hl_pct']= ((df['Adj. High']-df['Adj. Close'])/df['Adj. Close'])*100
df['pct_change']=((df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'])*100
df=df[['Adj. Close', 'hl_pct', 'pct_change', 'Adj. Volume']]
forecast_col='Adj. Close'
forecast_out=int(math.ceil(0.1*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
X=np.array(df.drop('label', 1))
y=np.array(df['label'])
X=preprocessing.scale(X)
y=np.array(df['label'])
X_train, X_test, y_train, y_test=model_selection.train_test_split(X, y, test_size=0.2)
clf=svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

