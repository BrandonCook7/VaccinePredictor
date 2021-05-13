# CS 315 Final Project by Brandon Cook and Matt Balint

import numpy as np
import pandas as pd

from draw_graph import graph_regression
from draw_graph import graph
from regression import calculate_regression

def get_states():
    return np.array(state_populations['State'].tolist())


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
def order_by_state():
    num_rows = vaccine_data.shape[0]
    state = vaccine_data.at[0,'location']
    start_index = 0
    temp_df = pd.DataFrame()

    for i in range(num_rows):
        current_state = vaccine_data.at[i, 'location']
        if (current_state != state):
            temp_df = vaccine_data.loc[start_index:i-1]

            state_vaccinations.update({vaccine_data['location'][i - 1] : temp_df})
            start_index = i
        state = current_state
    
    # Need to add the last state in the list since the for loop won't cover it.
    state_vaccinations.update({vaccine_data['location'][start_index] : temp_df})


def get_percent_vaccinated(date, state): # returns the % of people vaccinated in that state on that particular date
    state_data = state_vaccinations.get(state)
    date_data = state_data.loc[state_data['date'] == date] # Gets row of date from dataframe

    num_vaccinated = date_data.iloc[0]['people_fully_vaccinated'] # Gets value of vaccinated from row

    state_data = state_populations[state_populations['State'] == state]
    population = state_data.iloc[0]['Population_Estimate']
    return num_vaccinated/population


def get_state_input():
    state = ""
    while True:
        print("\nPlease enter a US state: ")
        state = input()
        state = state.title()
        if state == "New York":
            print("\Invalid state name. Please enter \'New York State\'")
        elif state not in states:
            print("\nInvalid state name. Please enter a valid US state like Washington.")
        else:
            break

    return state


def menu():
    choice = 0
    while choice <= 0 or choice > 5:
        print("--------US Covid-19 Vaccine Tracker --------\n")
        print("Enter a number below to retrieve a menu.\n")
        print("1. US Vaccine Prediction Graph") 
        print("2. State Vaccine Prediction Graph")
        print("3. Predict time left till herd immunity reached")
        print("4. Quit")
        choice = int(input())

        if choice == 1:
            print()
            # calculate_us_regression()
        elif choice == 2:
            state = get_state_input()
            x, y, regression_line = calculate_regression(state, state_vaccinations)
            graph_regression(x, y, regression_line, state)

        elif choice == 3:
            state = get_state_input()
            # calculate_us_regression()
            # make prediction using regression

if __name__ == '__main__':
    # Read data
    raw_vaccine_data = pd.read_csv("datasets/us_state_vaccinations0507.csv") # This spreadsheet has Covid-19 data up until May 1st
    vaccine_data = raw_vaccine_data[['date','location','people_vaccinated','people_fully_vaccinated']].copy() # Filter out some of the extra unnecessary columns
    state_populations = pd.read_csv("datasets/2019_Census_Data_Population.csv")
    states = get_states() # create a list of the names of all 50 states
    # current_date = date.today()

    state_vaccinations = {}

    # Clean data
    remove_non_states()
    order_by_state()

    # Display menu
    menu()