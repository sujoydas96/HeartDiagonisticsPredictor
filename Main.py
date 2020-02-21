import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as smf
from sklearn.linear_model import LinearRegression

#Dataset Import 
Dataset = pd.read_csv("Heart_Disease.csv")
Y = Dataset.get("Target")
Y = pd.DataFrame(Y,columns=['Target'])
X = Dataset.drop(['Target'], axis="columns")

#encoding Sex and avoiding dummy trap
Dum = pd.get_dummies(Dataset.Sex)
Dum.columns = ['Female','Male']
X = pd.concat([X,Dum],axis='columns')
X = X.drop(['Sex','Male'],axis="columns")

#encoding ChestPain and avoiding dummy trap
Dum = pd.get_dummies(Dataset.ChestPain)
Dum.columns = ['CPain_TypicalAngina','CPain_AtypicalAngina','CPain_NonAnginalPain','CPain_Asymptotic']
X = pd.concat([X,Dum],axis='columns')
X = X.drop(['ChestPain','CPain_AtypicalAngina'],axis="columns")

#Encoding Fasting Blood Sugar and avoiding dummy trap
Dum = pd.get_dummies(Dataset.Fasting_Blood_Sugar)
Dum.columns = ['Low_Fast_BSugar','Gre_Fast_BSugar']
X = pd.concat([X,Dum],axis='columns')
X = X.drop(['Fasting_Blood_Sugar','Gre_Fast_BSugar'],axis="columns")

#encoding Resting ECG and avoiding the dummy trap
Dum = pd.get_dummies(Dataset.Resting_ECG)
Dum.columns = ['ECG_Norm','ECG_ST_T','ECG_L_Ventri']
X = pd.concat([X,Dum],axis='columns')
X = X.drop(['Resting_ECG','ECG_Norm'],axis="columns")

#encoding Exercise Induced Angina and avoiding the dummy trap
Dum = pd.get_dummies(Dataset.Exercise_Induced_Angina)
Dum.columns = ['FAL_Ex_Ind_Ang','TRU_Ex_Ind_Ang']
X = pd.concat([X,Dum],axis='columns')
X = X.drop(['Exercise_Induced_Angina','FAL_Ex_Ind_Ang'],axis="columns")

#encoding Slope and avoiding the dummy trap
Dum = pd.get_dummies(Dataset.Slope)
Dum.columns = ['Slope_Upslope','Slope_Flat','Slope_downslope']
X = pd.concat([X,Dum],axis='columns')
X = X.drop(['Slope','Slope_Flat'],axis="columns")

#encoding Thalassemia and avoiding the dummy trap
Dum = pd.get_dummies(Dataset.Thalassemia)
Dum.columns = ['zero','one','two','three']
X = pd.concat([X,Dum],axis='columns')
X = X.drop(['Thalassemia','zero'],axis="columns")

#forming numpy dataframes
X = X.iloc[:,:].values
Y = Y.iloc[:,:].values

#multiple linear regression analysis ###########################################################
X = np.append(arr = np.ones((1025,1)).astype(int), values = X, axis = 1)

#significance level 0.05
X = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]] 
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]    
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]]    
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]]    
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]   
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]   
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14]]    
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,1,2,3,4,5,6,8,9,10,11,12,13]]    
Model = smf.OLS(endog = Y,exog = X).fit()

X = X[:,[0,1,2,3,4,5,6,7,8,10,11,12]]    
Model = smf.OLS(endog = Y,exog = X).fit()

#Train Test Split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#scaling the dependent matrix
SC_X = StandardScaler() 
X_Train = SC_X.fit_transform(X_Train)
X_Test = SC_X.fit_transform(X_Test)
SC_Y = StandardScaler()
Y_Train = SC_Y.fit_transform(Y_Train)

#################################End of Data Preprocessing###################################################
Regressor = LinearRegression().fit(X_Train,Y_Train)
Y_Pred = Regressor.predict(X_Test)

