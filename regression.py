import pandas as pd
import numpy as np
import datetime
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


#Converts a string to datetime then gregorian ordinal of the date, which helps the algorthim understand the data better
def convert_time(date_string):
    dt_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    ordNum = dt_object.toordinal()
    return ordNum

def calculate_regression(state, state_vaccinations_dict):
    state_data = state_vaccinations_dict.get(state)
    state_full_vaccinations = state_data['people_fully_vaccinated'].copy()
    state_full_vaccinations = state_full_vaccinations.interpolate()
    state_dates = state_data['date']
    state_dates_ord = []
    for i in state_dates: #Passing all the dates into function before splitting the data
        state_dates_ord.append(convert_time(i))

    ord_series = pd.Series(state_dates_ord)
    ord_series = ord_series.values.reshape(-1, 1)
    state_vaccinations = state_full_vaccinations.values.reshape(-1, 1)
    
    y = state_vaccinations
    x = ord_series

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    lin_reg = LinearRegression()
    lin_reg.fit(x,y)

    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(x)
    poly_reg.fit(x_poly,y)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(x_poly,y)

    temp = np.array(lin_reg2.predict(poly_reg.fit_transform(x)))
    y_temp = temp.flatten()

    return state_dates_ord, state_data['people_fully_vaccinated'].values, y_temp