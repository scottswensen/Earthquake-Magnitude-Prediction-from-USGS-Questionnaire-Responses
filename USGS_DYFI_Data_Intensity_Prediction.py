""" This script analyzes the data from USGS's "Did You Feel It" (DYFI)
Repository for all earthquakes in the contiguous United States and northern 
Mexico of magnitude 4.0 or larger over the past year. Each data entry includes
a listing of the number of people who reported feeling the motion to USGS in a 
specific town or jurisdiction, the Modified Mercalli Intensity (MMI) of the 
motion at the specified location calculated from the questionnaire responses, 
the longitude and latitude of the jurisdiction, as well as the distance to the 
epicenter of the earthquake and its magnitude. The purpose of this project is 
to see if an earthquake's magnitude (usually Moment Magnitude, Mw) at a given 
site can be predicted from the calculated intensity estimated from DYFI 
surveys submitted to USGS as well as the location of the responders and 
distance from the epicenter. Being able to estimate the moment magnitude from 
crowdsourced responses would be helpful in areas where seismic instrumentation 
is sparse and obtaining the earthquake magnitude quickly is difficult. The 
data was procured from this website for all earthquakes occurring in the 
contiguous United States with magnitudes of 4.0 or greater between 7/4/2017 
and 7/3/2018: https://earthquake.usgs.gov/data/dyfi/

A description of the Modified Mercalli Intensity (MMI) scale can be found here:
https://earthquake.usgs.gov/learn/topics/mercalli.php
"""    
# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import the data set
"""'M' is the moment magnitude of the event, 'MMI' is the Modified Mercalli 
intensity calculated using responses from those who reported feeling the event,
"Responses" give the number of people who reported feeling the event to USGS,
"Distance" gives the distance in kilometers from the responder and the 
earthquake epicenter, while the latitude and longitude columns give the 
location of the responder"""

df = pd.read_csv('DYFI.csv')

# Create another predictor which gives the portion of total event responses within each jurisdiction
resp_count = pd.DataFrame(df['Responses'].groupby(df['Event'], axis=0).sum())
df = df.join(resp_count, on = 'Event', rsuffix = '_total')

def res_portion (cols):
    responses = cols[0]
    responses_total = cols[1]
    return responses/responses_total
df['Pct_of_Resp'] = df[['Responses', 'Responses_total']].apply(res_portion, axis = 1)

df_reg = df[['M', 'MMI', 'Responses', 'Distance', 'Latitude', 'Longitude', 'Pct_of_Resp']]

# EXPLORE THE DATA VISUALLY
# Create pair plot of data
print('\nPair Plot of All Data Columns. Colors represent Magnitude:')
plt.figure()
sns.pairplot(df_reg, hue = 'M', palette = 'Blues')
plt.legend(loc='upper left', bbox_to_anchor=(-7, 7), ncol=2)
plt.show()

# Create joint plot - Distance v Magnitude
print('\nDistance [km] to Epicenter v. Magnitude:')
plt.figure()
sns.jointplot(x = 'Distance', y = 'M', data = df)
plt.show()

# Create joint plot - Distance v Magnitude
print('\nModified Mercalli Intensity (MMI) v. Magnitude:')
plt.figure()
sns.jointplot(x = 'MMI', y = 'M', data = df)
plt.show()

# Create joint plot - Responses v Magnitude
print('\nNumber of DYFI Responses v. Magnitude')
plt.figure()
sns.jointplot(x = 'Responses', y = 'M', data = df)
plt.show()

# Create joint plot - Portion of Responses v Magnitude
print('\nPortion of DYFI Responses in Response Area v. Magnitude')
plt.figure()
sns.jointplot(x = 'Pct_of_Resp', y = 'M', data = df)
plt.show()

# Separate into predictor and response arrays
X = df_reg[['Distance', 'Responses', 'MMI', 'Latitude', 'Longitude', 'Pct_of_Resp']]
y = df_reg['M']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# LINEAR REGRESSION MODEL
# Create the Linear Regression Training Model using all predictors
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# Create predictions and plot outcome
pred = lm.predict(X_test)
error_1 = ((np.array(y_test) - pred)**2).sum()
print('\n\nLINEAR REGRESSION RESULTS')
print('------------------------------------------------------------------')
print('\nFor linear regression, the root mean squared error is :', ('%.2E' % Decimal(error_1)), '.\n')

def truncate(f, n):
    '''Function truncates float'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

# Calculate mean error
abs_error = np.absolute(np.array(y_test) - pred)
abs_err_mean = abs_error.mean()
abs_err_median = np.median(abs_error)

abs_err_mean_pct = (abs_error/np.array(y_test)).mean()*100
abs_err_median_pct = np.median((abs_error/np.array(y_test)))*100

print('The mean magnitude error of this model is ', truncate(abs_err_mean,2), ' while the median magnitude error is ', truncate(abs_err_median,2), '.\n')
print('This represents a mean magnitude error of ', truncate(abs_err_mean/y_test.mean()*100,1),'% and a median magnitude error of ', truncate(abs_err_median_pct,1),'%')

# Plot results
plt.figure()
plt.scatter(y_test, pred)
plt.plot(range(0,8), ls = '--', c = 'k')
plt.xlabel('Actual Moment Magnitude')
plt.ylabel('Predicted Moment Magnitude')
plt.xlim(3, 7)
plt.ylim(3, 7)
plt.title('Linear Regression Results')
plt.show()

# POLYNOMIAL REGRESSION MODEL
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.transform(X_test)
poly_reg.fit(X_train, y_train)
lm_2 = LinearRegression()
lm_2.fit(X_train_poly, y_train)

# Create predictions and plot outcome of polynomial model
pred_poly = lm_2.predict(poly_reg.fit_transform(X_test))
error_poly = ((np.array(y_test) - pred_poly)**2).sum()
print('\n\nPOLYNOMIAL REGRESSION RESULTS')
print('------------------------------------------------------------------')
print('\nFor polynomial regression, the root mean squared error is :', ('%.2E' % Decimal(error_poly)), '.\n')

# Calculate mean error
abs_error_poly = np.absolute(np.array(y_test) - pred_poly)
abs_err_poly_mean = abs_error_poly.mean()
abs_err_poly_median = np.median(abs_error_poly)

abs_err_poly_mean_pct = (abs_error_poly/np.array(y_test)).mean()*100
abs_err_poly_median_pct = np.median((abs_error_poly/np.array(y_test)))*100

print('The mean magnitude error of this model is ', truncate(abs_err_poly_mean,2), ' while the median magnitude error is ', truncate(abs_err_poly_median,2), '.\n')
print('This represents a mean magnitude error of ', truncate(abs_err_poly_mean/y_test.mean()*100,1),'% and a median magnitude error of ', truncate(abs_err_poly_median_pct,1),'%')


# Plot Results
plt.figure()
plt.scatter(y_test, pred_poly)
plt.plot(range(0,8), ls = '--', c = 'k')
plt.xlabel('Actual Moment Magnitude')
plt.ylabel('Predicted Moment Magnitude')
plt.xlim(3, 7)
plt.ylim(3, 7)
plt.title('Polynomial Regression Results')
plt.show()

# SUPPORT VECTOR REGRESSION
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.transform(X_test)

# Fitting SVR to the dataset
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train_sc, y_train)

# Create predictions and plot outcome of SVR model
pred_svr = svr.predict(X_test_sc)
error_svr = ((np.array(y_test) - pred_svr)**2).sum()
print('\n\nSUPPORT VECTOR REGRESSION RESULTS')
print('------------------------------------------------------------------')
print('\nFor support vector regression, the root mean squared error is :', ('%.2E' % Decimal(error_svr)), '.\n')

# Calculate mean error
abs_error_svr = np.absolute(np.array(y_test) - pred_svr)
abs_err_svr_mean = abs_error_svr.mean()
abs_err_svr_median = np.median(abs_error_svr)

abs_err_svr_mean_pct = (abs_error_svr/np.array(y_test)).mean()*100
abs_err_svr_median_pct = np.median((abs_error_svr/np.array(y_test)))*100

print('The mean magnitude error of this model is ', truncate(abs_err_svr_mean,2), ' while the median magnitude error is ', truncate(abs_err_svr_median,2), '.\n')
print('This represents a mean magnitude error of ', truncate(abs_err_svr_mean/y_test.mean()*100,1),'% and a median magnitude error of ', truncate(abs_err_svr_median_pct,1),'%')

# Plot retults
plt.figure()
plt.scatter(y_test, pred_svr)
plt.plot(range(0,8), ls = '--', c = 'k')
plt.xlabel('Actual Moment Magnitude')
plt.ylabel('Predicted Moment Magnitude')
plt.xlim(3, 7)
plt.ylim(3, 7)
plt.title('Support Vector Regression Results')
plt.show()

# RANDOM FOREST REGRESSION
# Fitting SVR to the dataset
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 1000)
rfr.fit(X_train, y_train)

# Create predictions and plot outcome of SVR model
pred_rfr = rfr.predict(X_test)
error_rfr = ((np.array(y_test) - pred_rfr)**2).sum()
print('\n\nRANDOM FOREST REGRESSION RESULTS')
print('------------------------------------------------------------------')
print('\nFor random forest regression, the root mean squared error is :', ('%.2E' % Decimal(error_rfr)), '.\n')

# Calculate mean error
abs_error_rfr = np.absolute(np.array(y_test) - pred_rfr)
abs_err_rfr_mean = abs_error_rfr.mean()
abs_err_rfr_median = np.median(abs_error_rfr)

abs_err_rfr_mean_pct = (abs_error_rfr/np.array(y_test)).mean()*100
abs_err_rfr_median_pct = np.median((abs_error_rfr/np.array(y_test)))*100

print('The mean magnitude error of this model is ', truncate(abs_err_rfr_mean,3), ' while the median magnitude error is ', truncate(abs_err_rfr_median,3), '.\n')
print('This represents a mean magnitude error of ', truncate(abs_err_rfr_mean/y_test.mean()*100,2),'% and a median magnitude error of ', truncate(abs_err_rfr_median_pct,2),'%')

# Plot results
plt.figure()
plt.scatter(y_test, pred_rfr)
plt.plot(range(0,8), ls = '--', c = 'k')
plt.xlabel('Actual Moment Magnitude')
plt.ylabel('Predicted Moment Magnitude')
plt.xlim(3, 7)
plt.ylim(3, 7)
plt.title('Random Forest Regression Results')
plt.show()
print('The random forest resgression results are quite good and suggest that using reported crowdsourced data from DYFI questionnaires can be effective in accurately estimating earthquake magnitudes.')