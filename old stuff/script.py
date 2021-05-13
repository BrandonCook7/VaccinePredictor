# CS 315 Final Project by Brandon Cook and Matt Balint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from datetime import date
import datetime

def get_list_of_states():
    x = state_populations['State'].tolist()
    return np.array(x)

# Remove any US territories or prisons from the data set
def remove_non_states():
    x = vaccine_data['location'] # get locations column
    i = len(x) - 1 # number of rows

    while i > 0:
        if x[i] not in states:
            vaccine_data.drop(vaccine_data.index[i], inplace=True)
        i -= 1

    vaccine_data.reset_index(drop=True, inplace=True) # update the indexes so they match with the length of the DataFrame


# Group data together for each state and put it in a dictionary
def order_by_states():
    num_rows = vaccine_data.shape[0]
    state = vaccine_data.at[0,'location']
    start_index = 0
    state_index = 0
    tempDf = pd.DataFrame()

    for i in range(num_rows):
        current_state = vaccine_data.at[i, 'location']
        if (current_state != state):
            tempDf = vaccine_data.loc[start_index:i-1]

            state_vaccinations_dict.update({vaccine_data['location'][i - 1] : tempDf})
            start_index = i
            state_index += 1
        state = current_state
    
    # Need to add the last state in the list since the for loop won't cover it.
    state_vaccinations_dict.update({vaccine_data['location'][start_index] : tempDf})

    for i, data in enumerate(state_vaccinations):
        state_vaccinations_dict.update({states[i] : data})

    print("test")

def get_percent_vaccinated(date, state): # returns the % of people vaccinated in that state on that particular date
    state_data = state_vaccinations_dict.get(state)
    date_data = state_data.loc[state_data['date'] == date] # Gets row of date from dataframe

    num_vaccinated = date_data.iloc[0]['people_fully_vaccinated'] # Gets value of vaccinated from row

    state_data = state_populations[state_populations['State'] == state]
    population = state_data.iloc[0]['Population_Estimate']
    return num_vaccinated/population

def graph_vaccinated(date_until, state):
    start_date = date(2021, 1, 12)
    end_date = date_until
    date_range = pd.date_range(start_date, end_date)
    vaccination_percents = []
    date_list = []
    for index_date in date_range:
        x = get_percent_vaccinated(index_date.strftime("%Y-%m-%d"), state)
        vaccination_percents.append(x)
        date_list.append(index_date.strftime("%Y-%m-%d"))
    plt.plot(vaccination_percents)
    plt.show()


#Converts a string to datetime then gregorian ordinal of the date, which helps the algorthim understand the data better
def convert_time(date_string):
    dt_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    ordNum = dt_object.toordinal()
    return ordNum

def regressionAlgo(state):
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

    x_grid = np.arange(min(x),max(x),0.1)
    x_grid = x_grid.reshape((len(x_grid),1))
    plt.scatter(x,y,color='red')
    plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')

    plt.title('Polynomial Regression')
    plt.xlabel('Date')
    plt.ylabel('Total Fully Vaccinated')
    plt.show()

def get_state_input():

    state = ""
    while True:
        print("Please enter a US state: ")
        state = input()
        state = state.title()
        if state not in states:
            print("Invalid state name. Please enter a valid US state like Washington.")
        else:
            break

    return state

def calculate_us_prediction():
    herd_immunity = 0.8 # assume 80% is needed to achieve herd immunity

    for state in states:
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

def menu():
    choice = 0
    while choice <= 0 or choice > 5:
        print("--------US Covid-19 Vaccine Tracker --------\nEnter a number below to retrieve a menu.\n")
        print("1. US Vaccine Prediction Graph") 
        print("2. State Vaccine Prediction Graph")
        print("3. Predict time left till herd immunity reached")
        print("4. Quit")
        choice = int(input())

        if choice == 1:
            print()
            # calculate_us_prediction()
        elif choice == 2:
            state = get_state_input()
            regressionAlgo(state)
        elif choice == 3:
            state = get_state_input()


# Read data
raw_vaccine_data = pd.read_csv("datasets/us_state_vaccinations0507.csv") # This spreadsheet has Covid-19 data up until May 1st
vaccine_data = raw_vaccine_data[['date','location','people_vaccinated','people_fully_vaccinated']].copy() # Filter out some of the extra unnecessary columns
state_populations = pd.read_csv("datasets/2019_Census_Data_Population.csv")
states = get_list_of_states() # create a list of the names of all 50 states
current_date = date.today()

state_vaccinations = []
state_vaccinations_dict = {}

# Clean data
remove_non_states()
order_by_states()

# Display menu
menu()