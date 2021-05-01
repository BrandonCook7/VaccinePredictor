import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from datetime import date

unordered_vacc_df = pd.read_csv("datasets/us_state_vaccinations.csv")
state_vacc = [pd.DataFrame]
us_state_pop = pd.read_csv("datasets/2019_Census_Data_Population.csv")
#print(unordered_vacc_df)
vacc_df = unordered_vacc_df[['date','location','total_vaccinations','total_distributed','people_vaccinated','people_fully_vaccinated', 'daily_vaccinations']].copy()

current_date = date.today()


def removeNonStates():
    stateNames = us_state_pop['State']
    stateList = []
    for i in range(len(stateNames)):
        stateList.append(stateNames[i])
    print(stateList)
    vacc_df_temp = vacc_df
    temp_location = vacc_df_temp['location']
    i = len(temp_location)-1
    print(vacc_df)
    
    while i > 0:
        try:
            index = stateList.index(temp_location[i])
        except ValueError:
            print(vacc_df['location'][i])

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
            tempDf = vacc_df.loc[startOfLoc:i]
            startOfLoc = i
            
            state_vacc.append(tempDf)
        currentLoc = vacc_df['location'][i]
            


#print(us_state_pop)
#print(state_vacc)
print('hello')
#print(current_date)
removeNonStates()
orderByStates()

print(state_vacc)