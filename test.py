import inline as inline
import matplotlib

import xgboost
import pandas as pd
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
'''
originaldata = pd.read_csv('train.csv')
count = originaldata.shape[0]
data_train = originaldata.iloc[:int(count*0.7)]
data_test = originaldata.iloc[int(count*0.7+1):]
#data_train, data_test = train_test_split(originaldata, test_size=0.3, random_state=42)
'''

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')


#print(data_train)
print(data_train['SalePrice'].describe())
sns.distplot(data_train['SalePrice'])

#skewness and kurtosis
print("Skewness: %f" % data_train['SalePrice'].skew())
print("Kurtosis: %f" % data_train['SalePrice'].kurt())

#Exploratory Visualization
plt.figure(figsize=(15,8))
sns.boxplot(data_train.YearBuilt, data_train.SalePrice)



plt.figure(figsize=(12,6))
plt.scatter(x=data_train.GrLivArea, y=data_train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)

data_train.drop(data_train[(data_train["GrLivArea"]>4000)&(data_train["SalePrice"]<300000)].index,inplace=True)
full=pd.concat([data_train,data_test], ignore_index=True)
'''
这里我们先将训练数据集和测试数据集合并为一个数据集，这样做除了方便之后可以同时对训练数据集和测试数据集进行数据清洗和特征工程，此外，也考虑在之后对类别型变量（category variable）需要进行标签编码（LabelEncoder）和独热编码(OneHotEncoder），而标签编码和独热编码主要是基于类别变量的特征值进行编码，为了避免测试集的类别变量存在训练集所不具有的特征值，而影响模型的性能，因此这里先将两个数据集进行合并，在最后对模型进行训练时再将合并的数据集按照索引重新分割为训练集和测试集。
'''
full.drop(['Id'],axis=1, inplace=True)
full.shape

plt.show()

#Data Cleaning
'''
对于缺失数据的处理，通常会有以下几种做法

如果缺失的数据过多，可以考虑删除该列特征
用平均值、中值、分位数、众数、随机值等替代。但是效果一般，因为等于人为增加了噪声
用插值法进行拟合
用其他变量做预测模型来算出缺失变量。效果比方法1略好。有一个根本缺陷，如果其他变量和缺失变量无关，则预测的结果无意义
最精确的做法，把变量映射到高维空间。比如性别，有男、女、缺失三种情况，则映射成3个变量：是否男、是否女、是否缺失。缺点就是计算量会加大。
首先观察一下各列数据缺失的情况（篇幅所限并未列出所有的缺失列）
'''
aa = full.isnull().sum()

print(aa[aa>0].sort_values(ascending=False))
corrmat = full.corr()#xiangguanxishu
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
#print(full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count']))

full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])

full["LotAreaCut"] = pd.qcut(full.LotArea,10)
full.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])

full['LotFrontage']=full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

full['LotFrontage']=full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full[col].fillna(0, inplace=True)

cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    full[col].fillna("None", inplace=True)

cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual",
             "SaleType", "Exterior1st", "Exterior2nd"]
for col in cols2:
        full[col].fillna(full[col].mode()[0], inplace=True)

full.isnull().sum()[full.isnull().sum()>0]

NumStr = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","KitchenAbvGr","MoSold","YrSold","YearBuilt","YearRemodAdd","LowQualFinSF","GarageYrBlt"]
for col in NumStr:
    full[col]=full[col].astype(str)
full.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count'])

#sort by the median
#about full function in numpy: https://www.cjavapy.com/article/86/
def map_values():
    full["oMSSubClass"] = full.MSSubClass.map({'180': 1,
                                               '30': 2, '45': 2,
                                               '190': 3, '50': 3, '90': 3,
                                               '85': 4, '40': 4, '160': 4,
                                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                               '120': 6, '60': 6})

    full["oMSZoning"] = full.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV': 1,
                                                   'IDOTRR': 2, 'BrDale': 2,
                                                   'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                   'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                   'NPkVill': 5, 'Mitchel': 5,
                                                   'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                   'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                   'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                   'StoneBr': 9,
                                                   'NoRidge': 10, 'NridgHt': 10})

    full["oCondition1"] = full.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    full["oBldgType"] = full.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    full["oExterior1st"] = full.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    full["oMasVnrType"] = full.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    full["oExterQual"] = full.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFoundation"] = full.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    full["oBsmtQual"] = full.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oBsmtExposure"] = full.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    full["oHeating"] = full.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    full["oHeatingQC"] = full.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oKitchenQual"] = full.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFunctional"] = full.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    full["oFireplaceQu"] = full.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oGarageType"] = full.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    full["oGarageFinish"] = full.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    full["oPavedDrive"] = full.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    full["oSaleType"] = full.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    full["oSaleCondition"] = full.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

    return "Done!"

map_values()
#function drop on pandas: https://www.cnblogs.com/demo-deng/p/9609824.html
full.drop("LotAreaCut",axis=1,inplace=True)
full.drop(['SalePrice'],axis=1,inplace=True)

#https://www.jianshu.com/p/d6877c36e977
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X

#skew_dummies:这是实现将数据正态分布化,然后整体One-Hot化的代码
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew=0.5):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        X = pd.get_dummies(X)
        return X

# build pipeline
pipe = Pipeline([
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1)),
    ])

# save the original data for later use
full2 = full.copy()

data_pipe = pipe.fit_transform(full2)

data_pipe.shape

data_pipe.head()

#function RobustScaler https://blog.csdn.net/qq_33472765/article/details/85944256
'''
如果你的数据包含许多异常值，使用均值和方差缩放可能并不是一个很好的选择。这种情况下，你可以使用 robust_scale 以及 RobustScaler 作为替代品。它们对你的数据的中心和范围使用更有鲁棒性的估计。
'''
scaler = RobustScaler()

n_train=data_train.shape[0]

X = data_pipe[:n_train]
test_X = data_pipe[n_train:]
y= data_train.SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(data_train.SalePrice)
test_X_scaled = scaler.transform(test_X)
#Lasso algorithm:https://baike.baidu.com/item/LASSO/20366865?fr=aladdin
lasso=Lasso(alpha=0.001)
lasso.fit(X_scaled,y_log)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=data_pipe.columns)

FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
print(FI_lasso)
plt.show()


class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.additional == 1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]

            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
                "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]

            return X

pipe = Pipeline([
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1)),
    ])

#PCA https://baike.baidu.com/item/pca%E6%8A%80%E6%9C%AF/10408698?fr=aladdin

full_pipe = pipe.fit_transform(full)#先拟合数据，然后转化它将其转化为标准形式，一般应用在训练集中。https://blog.csdn.net/zaishijizhidian/article/details/79949379
full_pipe.shape#shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数。c.shape[1] 为第一维的长度，c.shape[0] 为第二维的长度。https://www.cnblogs.com/bnuvincent/p/7754932.html
n_train=data_train.shape[0]
#print(n_train)
X = full_pipe[:n_train]
#print(X)
test_X = full_pipe[n_train:]
#print(test_X)
y= data_train.SalePrice
#print(y)

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(data_train.SalePrice)
test_X_scaled = scaler.transform(test_X)

pca = PCA(n_components=410)

X_scaled=pca.fit_transform(X_scaled)
test_X_scaled = pca.transform(test_X_scaled)

#print(X_scaled.shape, test_X_scaled.shape)

#Modeling & Evaluation
#rmse: Root Mean Square Error:https://blog.csdn.net/genghaihua/article/details/81119618
#cross_val_score function in sklearn:https://www.cnblogs.com/lzhc/p/9175707.html
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),XGBRegressor()]

names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]
'''
zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。

我们可以使用 list() 转换来输出列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
https://www.runoob.com/python3/python3-func-zip.html
'''
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, y_log)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))


class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])

grid(Lasso()).grid_get(X_scaled, y_log, {'alpha': [0.0004, 0.0005, 0.0007, 0.0009], 'max_iter': [10000]})
grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[35,40,45,50,55,60,65,70,80,90]})
grid(SVR()).grid_get(X_scaled,y_log,{'C':[11,13,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})

param_grid={'alpha':[0.2,0.3,0.4], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1]}
grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)
grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3],'max_iter':[10000]})

#Ensemble Methods
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.sum(single))
        return w

lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()

w1 = 0.02
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.03
w6 = 0.2

weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])
score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())
weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])

score = rmse_cv(weight_avg,X_scaled,y_log)
print(score.mean())


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)

    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))
        test_single = np.zeros((test_X.shape[0], 5))
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                oof[val_index, i] = clone_model.predict(X[val_index])
                test_single[:, j] = clone_model.predict(test_X)
            test_mean[:, i] = test_single.mean(axis=1)
        return oof, test_mean

# must do imputer first, otherwise stacking won't work, and i don't know why.
a = Imputer().fit_transform(X_scaled)
b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()

stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)

score = rmse_cv(stack_model,a,b)
print(score.mean())

X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)

X_train_stack.shape, a.shape

X_train_add = np.hstack((a,X_train_stack))

X_test_add = np.hstack((test_X_scaled,X_test_stack))
X_train_add.shape, X_test_add.shape
score = rmse_cv(stack_model,X_train_add,b)
print(score.mean())

stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
stack_model.fit(a,b)

pred = np.exp(stack_model.predict(test_X_scaled))
result=pd.DataFrame({'Id':data_test.Id, 'SalePrice':pred})
result.to_csv("submission.csv",index=False)

weight_avg = AverageWeight(mod = [svr,ker],weight=[0.5,0.5])
weight_avg.fit(a,b)
pred_weight_2model= np.exp(weight_avg.predict(test_X_scaled))
result=pd.DataFrame({'Id':data_test.Id, 'SalePrice':pred_weight_2model})
result.to_csv("submission_2model.csv",index=False)