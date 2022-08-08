#This file is to convert the updated user data into a file for easy importing later on

import pandas as pd
import time
import json
import ast
import numpy as np
from scipy.sparse.linalg import svds
import math
from sklearn.impute import KNNImputer
import sys
from sklearn.model_selection import train_test_split
import json, time, sys
from alive_progress import alive_bar

#colors red,yellow,green,blue,purple,violet,Bold,Underline,End
class c:
    r = '\033[91m'
    y = '\033[93m'
    g = '\033[92m'
    b = '\033[96m'
    p = '\033[94m'
    v = '\033[95m'
    B = '\033[1m'
    U = '\033[4m'
    E = '\033[0m'


start_time = time.time()
print(f'Importing original...')
original = pd.read_csv('users.csv.gz').drop('Unnamed: 0',axis=1)
print(f'{round(time.time()-start_time)}s. Importing updated...')
start_time = time.time()
updated = pd.read_csv('updated_users.csv.gz').drop('Unnamed: 0',axis=1)
print(f'{round(time.time()-start_time)}s. Done Importing. Making differences_dict')

differences = dict()
#key = _id, value = dict(key=anime,value=rating)

with alive_bar(len(updated),bar='bubbles',spinner=None,dual_line=True) as bar:

    #for each user in updated
    for _,row in updated.iterrows():
        bar()

        #find their old and new ratings
        new_ratings = ast.literal_eval(row['ratings'])
        old_ratings = ast.literal_eval(original[original['_id']==row['_id']].iloc[0]['ratings'])

        #convert them into dictionaries for O(1) search
        old_ratings_dict = dict()
        for x in old_ratings:
            old_ratings_dict[x['animeID']] = x['rating']

        differences_dict = dict()
        for x in new_ratings:
            #if its an anime never seen before, and not 0
            if x['animeID'] not in old_ratings_dict and x['rating'] != 0:
                differences_dict[x['animeID']] = x['rating']
        
        if len(differences_dict) != 0:
            differences[row['_id']] = differences_dict

with open('test_ratings.txt','w') as file:
    file.write(str(differences))

print('Done')
