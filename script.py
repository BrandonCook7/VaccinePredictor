import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from datetime import date

unordered_vacc_df = pd.read_csv("datasets/us_state_vaccinations.csv")
state_vacc = [pd.DataFrame]
state_vacc_dict = {}
us_state_pop = pd.read_csv("datasets/2019_Census_Data_Population.csv")
#print(unordered_vacc_df)
vacc_df = unordered_vacc_df[['date','location','total_vaccinations','total_distributed','people_vaccinated','people_fully_vaccinated', 'daily_vaccinations']].copy()

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

graphVaccinated(date(2021, 4, 20), "New Hampshire")

