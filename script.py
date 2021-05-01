import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

unordered_vacc_df = pd.read_csv("us_state_vaccinations.csv")
state_vacc = [pd.DataFrame]
#print(unordered_vacc_df)
vacc_df = unordered_vacc_df[['date','location','total_vaccinations','total_distributed','people_vaccinated','people_fully_vaccinated', 'daily_vaccinations']].copy()

#print(vacc_df)
#vacc_df.info()

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
            
           
            print(tempDf)
            state_vacc.append(tempDf)
        currentLoc = vacc_df['location'][i]
            

orderByStates()
print('hello')   
    