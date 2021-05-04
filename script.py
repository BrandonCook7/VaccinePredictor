import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import matplotlib.pyplot as plt
from datetime import date
import datetime

unordered_vacc_df = pd.read_csv("datasets/us_state_vaccinations0501.csv")
state_vacc = [pd.DataFrame]
state_vacc_dict = {}
us_state_pop = pd.read_csv("datasets/2019_Census_Data_Population.csv")
#print(unordered_vacc_df)
#vacc_df = unordered_vacc_df[['date','location','total_vaccinations','total_distributed','people_vaccinated','people_fully_vaccinated', 'daily_vaccinations']].copy()
vacc_df = unordered_vacc_df[['date','location','people_vaccinated','people_fully_vaccinated']].copy()
current_date = date.today()



def removeNonStates():
    stateNames = us_state_pop['State']
    stateList = []
    for i in range(len(stateNames)):
        stateList.append(stateNames[i])
    #print(stateList)
    vacc_df_temp = vacc_df
    temp_location = vacc_df_temp['location']
    i = len(temp_location)-1
    #print(vacc_df)
    
    while i > 0:
        try:
            index = stateList.index(temp_location[i])
        except ValueError:
            #print(vacc_df['location'][i])

            vacc_df.drop(vacc_df.index[i], inplace=True,axis=0)#Inplace keeps the orginal without needing a copy, axis=0 means to go by rows
        i -= 1

    vacc_df.reset_index(drop=True, inplace=True)#Drop updates the indexes so they match with the length of the DataFrame






def orderByStates():#This splits up the main dataframe into smaller dataframes nested in a Python Array, they are divided by their location name
    num_vacc_rows = vacc_df.shape[0]
    currentLoc = ''
    tempDf = pd.DataFrame()
    startOfLoc = 0
    for i in range(num_vacc_rows):
        if (vacc_df['location'][i] != currentLoc) and (i != 0):
            del tempDf
            tempDf = pd.DataFrame()
            tempDf = vacc_df.loc[startOfLoc:i-1]
            startOfLoc = i
            
            state_vacc.append(tempDf)
            state_vacc_dict.update({vacc_df['location'][i-1] : tempDf})
        currentLoc = vacc_df['location'][i]
            
def getPercVacc(date, state):#Given a data and state it returns the percent of people vaccinated in that state on that date
    state_data = state_vacc_dict.get(state)#Gets state Dataframe
    date_data = state_data.loc[state_data['date'] == date]#Gets row of date from dataframe
    vacc_on_date = date_data.iloc[0]['people_fully_vaccinated']#Gets value of vaccinated from row

    state_data = us_state_pop[us_state_pop['State'] == state]
    pop_of_state = state_data.iloc[0]['Population_Estimate']
    percVaccinated = vacc_on_date/pop_of_state


    #print(vacc_on_date)
    #print(pop_of_state)
    #print(percVaccinated)
    return percVaccinated

def graphVaccinated(dateUntil, state):
    start_date = date(2021, 1, 12)
    end_date = dateUntil
    date_range = pd.date_range(start_date, end_date)
    vaccList = []
    dateList = []
    for index_date in date_range:
        vaccPerc = getPercVacc(index_date.strftime("%Y-%m-%d"), state)
        vaccList.append(vaccPerc)
        dateList.append(index_date.strftime("%Y-%m-%d"))
    plt.plot(vaccList)
    plt.show()


removeNonStates()
orderByStates()
def convertTime(dateString):#Converts a string to datetime then gregorian ordinal of the date, which helps the algorthim understand the data better
    dt_object = datetime.datetime.strptime(dateString, "%Y-%m-%d")
    ordNum = dt_object.toordinal()
    return ordNum

def regressionAlgo(state):
    state_data = state_vacc_dict.get(state)
    
    state_data.reset_index(inplace=True)
    #state_vaccinations = state_data[['people_fully_vaccinated', 'people_vaccinated']].copy()
    state_vaccinations = state_data['people_fully_vaccinated'].copy()
    print(state_vaccinations)
    state_vaccinations = state_vaccinations.interpolate()
    print(state_vaccinations)
    state_dates = state_data['date']
    state_dates_ord = []
    for i in state_dates: #Passing all the dates into function before splitting the data
        state_dates_ord.append(convertTime(i))


    ord_series = pd.Series(state_dates_ord)
    ord_series = ord_series.values.reshape(-1, 1)
    #state_vaccinations = state_vaccinations.reset_index()
    print(state_vaccinations)
    y = state_vaccinations
    x = ord_series
    print(x)
    #y = ord_series.values.astype(np.float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    #x_train= x_train.values.reshape(-1,1)
    #x_test = x_test.values.reshape(-1,1)

    logistic_regr = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_regr.fit(x_train, y_train)

    prediction = logistic_regr.predict(x_test)
    print("Prediction: ", prediction)
    score = logistic_regr.score(x_test,y_test)
    print("Score: ", score)

    #score = logistic_regr.score(y_test, y_test)
    #print(score)
regressionAlgo("New Hampshire")
#graphVaccinated(date(2021, 5, 1), "New Hampshire")
#print(convertTime("2021-04-05"))
#print(vacc_df)