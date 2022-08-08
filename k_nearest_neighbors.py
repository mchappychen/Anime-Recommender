#This file uses KNN to predict ratings for a user

### CONFIG
knn_k = 5 #number of neighbors for KNN Imputer
user_rows = 100 #number of users to get. set to None for all
### END CONFIG

from alive_progress import alive_bar
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
import json, time, sys, requests, os

# pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', 20)

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

#returns user_data,anime_data,ratings_pivot
def get_data(user=None):

    print(f'{c.b}Getting a random sample of {c.p}{user_rows}{c.b} users from {c.v}users.csv.gz{c.b} and all animes from {c.v}animes.csv.gz{c.b}...{c.E}')
    start_time = time.time()

    user_data = pd.read_csv('users.csv.gz').sample(user_rows).drop('Unnamed: 0',axis=1)
    anime_data = pd.read_csv('animes.csv.gz').drop('Unnamed: 0',axis=1)

    #Get user's MAL 
    if user:
        access_token = 'GET YOUR OWN ACCESS TOKEN'
        response = requests.get(
            f'https://api.myanimelist.net/v2/users/{user}/animelist?limit=1000&fields=list_status&nsfw=true', 
            headers = {'Authorization':f'Bearer {access_token}'}
        )
        try:
            data = json.loads(response.text)['data']   
        except Exception as e:
            print(f'{c.r}Error: User {c.v}{user}{c.r} does not have a MAL{c.E}')
            sys.exit()
        ratings = []
        for rating in json.loads(response.text)['data']:
            ratings.append({
                'animeID' : rating['node']['id'],
                'rating' : rating['list_status']['score'],
                'status' : rating['list_status']['status']
            })
        user_data = pd.concat([user_data,pd.DataFrame([{'_id':user,'ratings':str(ratings)}])])
    
    #extract ratings from each user
    print(f'{c.b}Extracting user ratings...{c.E}')
    ratings = []
    for _,row in user_data.iterrows():
        for anime in ast.literal_eval(row['ratings']):
            #skip the unrated animes
            if anime['rating'] == 0:
                continue
            ratings.append({
                'user_id':row['_id'],
                'anime_id':anime['animeID'],
                'rating':anime['rating']})

    #some users may have not had any anime ratings, they're skipped

    #Format so columns are animes, and rows are users                                                                                                                                                                                                                
    ratings_pivot = pd.DataFrame(ratings).pivot(index = 'user_id', columns ='anime_id', values = 'rating')

    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}s{c.b}. Sample list of users: {c.v}{user_data.sample(3)["_id"].tolist()}{c.E}')
    
    return user_data,anime_data,ratings_pivot

#returns RMSE for a user's predictions, and `n`, number of predictions
def get_user_RMSE(user,preds,user_data,verbose=False):
    if verbose:
        print(f' {c.b}Calculating Root-Mean-Square-Error for {c.p}{user}{c.b}...{c.E}')

    #Get the predicted ratings of the user
    transposed = preds.loc[preds['user_id'] == user].drop('user_id',axis=1).transpose()
    transposed.rename(columns={transposed.columns[0]:'rating'},inplace=True)

    #arrange the predicted ratings from best to worst
    user_predictions = transposed.sort_values('rating',ascending=False).reset_index()

    #Get the user's actual ratings
    user_rating_list = user_data.loc[user_data['_id'] == user]['ratings']
    user_rating_list = pd.DataFrame([{'anime_id':x['animeID'],'rating':x['rating']} for x in ast.literal_eval(user_rating_list.values[0])])

    #The code above can be re-used

    #Keep only the predicted ratings for animes that the user has watched so we can compare RMSE
    predicted = user_predictions.drop(user_predictions[~user_predictions['anime_id'].isin(user_rating_list['anime_id'])].index)

    #For some reason rows in right matrix that don't exist in left matrix get added. This left-join isn't perfect unless we drop NA rows
    predicted = pd.merge(user_rating_list,predicted,how='left',on='anime_id',suffixes=('_actual','_predicted')).dropna()
 
    #Remove where actual ratings = 0 since the user hasn't rated them
    predicted = predicted[predicted['rating_actual'] != 0]

    #This shouldn't run since we filtered out all ratings=0
    if len(predicted) == 0:
        print(f'  {c.r}ERROR: User {c.v}{user}{c.y} had an unrated anime in their prediction!{c.E}')
        return 0,0

    #Calculate RMSE
    RMSE = math.sqrt(sum((x['rating_predicted']-x['rating_actual'])**2 for _,x in predicted.iterrows())/len(predicted))
    return RMSE,len(predicted)

#returns overall RMSE
def get_overall_rmse(preds,user_data):
    # print(f'{c.b}Getting overall RMSE...{c.E}')
    RMSE = 0
    N = 0
    for _,x in preds.iterrows():
        x_rmse,n = get_user_RMSE(x['user_id'],preds,user_data,verbose=False)
        RMSE += (x_rmse**2)*n
        N += n
    return math.sqrt((RMSE/n))

#Get a list of anime recommendations for a user
def get_user_recommendations(user,amount,preds,anime_data):
    print(f'{c.b}Getting top {c.p}{amount}{c.b} recommendations for {c.p}{user}{c.b}...{c.E}')

    #Get the predicted ratings of the user
    transposed = preds.loc[preds['user_id'] == user].drop('user_id',axis=1).transpose()
    transposed.rename(columns={transposed.columns[0]:'rating'},inplace=True)

    #Sort the predicted ratings
    user_predictions = transposed.sort_values('rating',ascending=False).reset_index()

    #Get the ratings the user has rated
    user_rating_list = user_data.loc[user_data['_id'] == user]['ratings']
    user_rating_list = pd.DataFrame([{'anime_id':x['animeID'],'rating':x['rating']} for x in ast.literal_eval(user_rating_list.values[0])])


    print(len(user_rating_list))
    #Predictions - user ratings = recommendations
    recommendations = user_predictions.drop(user_predictions[user_predictions['anime_id'].isin(user_rating_list['anime_id'])].index)

    #Get anime information, and rename `ratings` column to avoid confusion
    recommendations = pd.merge(recommendations,anime_data,how='left',left_on='anime_id',right_on='_id').dropna().rename(columns={'rating_x':'predicted_rating','rating_y':'MAL_overall_rating'})

    return recommendations.head(amount)

def single_knn(matrix,k,user):

    print(f'{c.b}Doing KNN for user {c.p}{user}{c.b}...{c.g}')

    if k<1 or not isinstance(k,int):
        raise Exception('k must be natural number')

    new_matrix = matrix.copy(deep=True)

    with alive_bar(len(matrix.columns),bar='bubbles',spinner=None,dual_line=True) as bar:

        #loop through each column, check for nan
        for index,value in enumerate(matrix.loc[user]):
            bar()
            if not np.isnan(value):
                continue

            #(user,distance)
            distances = []

            #calculate the distance to every other row
            for user2,ratings2 in matrix.iterrows():

                #make sure there's a value in index column
                if np.isnan(ratings2.iloc[index]):
                    continue

                #make sure we're not repeating our own row
                if user == user2:
                    continue
                
                #keep track of how many nans there are
                nans = 0
                total_ratings = len(ratings2)
                distance_sum = 0
                
                #go through each rating from left to right
                for index2 in range(len(ratings2)):

                    #check if either rating is a nan
                    if np.isnan(ratings2.iloc[index2]) or np.isnan(ratings2.iloc[index2]):
                        nans += 1
                        continue

                    #calculate weighted manhattan distance to other rows
                    distance_sum += abs(ratings2.iloc[index2] - ratings2.iloc[index2])
                
                #if they're all nans, set distance to -1 so we can filter them out later
                if nans == total_ratings:
                    distances.append((user2,-1))
                else:
                    #calculate weighted distance (perhaps test it without weights too?)
                    distance = (total_ratings/(total_ratings - nans)) * distance_sum

                    #store manhattan distance along with its user
                    distances.append((user2,distance))
        
            #find the top k users with the smallest distance, ignore -1s
            similar_users = []
            distances = sorted(distances,key=lambda x: x[1])
            
            if k > len(distances):
                k = len(distances)
            
            #only focus on valid distances
            for x in range(k):
                if distances[x][1] == -1:
                    continue
                similar_users.append(distances[x][0])

            #if there aren't any valid, raise an error (this shouldn't happen)
            if len(distances) == 0:
                raise Exception(f'ERROR, distances is empty: {distances}')

            #in the rare occasion that all distances are -1, pick a user to impute
            if len(similar_users) == 0:
                similar_users.append(distances[0][0])

                #temporarily set these all to 0
                new_matrix.loc[user].iloc[index] = 0
                continue

            #impute that column with the average
            column_average = [int(matrix.loc[x].iloc[index]) for x in similar_users]

            #we use new_matrix instead of the original one it doesn't influence KNN
            new_matrix.loc[user].iloc[index] = sum(column_average)/len(similar_users)
            

    #reset print color
    print(f'{c.E}')

    #set row index as column user_id
    new_matrix['user_id'] = new_matrix.index
    return new_matrix.loc[user]

#predict entire matrix
def full_knn(matrix,k):

    print(f'{c.b}Doing full matrix KNN...{c.g}')

    if k<1 or not isinstance(k,int):
        raise Exception('k must be natural number')

    #create new imputed matrix
    new_matrix = matrix.copy(deep=True)

    with alive_bar(len(matrix),bar='bubbles',spinner=None,dual_line=True) as bar:

        #for each row
        for user1,ratings1 in matrix.iterrows():

            # print(f'{c.b}Doing KNN for user {c.p}{user1}{c.b}...{c.E}')

            #loop through each column, check for nan
            for index,value in enumerate(ratings1):
                if not np.isnan(value):
                    continue

                #(user,distance)
                distances = []

                #calculate the distance to every other row
                for user2,ratings2 in matrix.iterrows():

                    #make sure there's a value in index column
                    if np.isnan(ratings2.iloc[index]):
                        continue

                    #make sure we're not repeating our own row
                    if user1 == user2:
                        continue
                    
                    #keep track of how many nans there are
                    nans = 0
                    total_ratings = len(ratings1)
                    distance_sum = 0
                    
                    #go through each rating from left to right
                    for index2 in range(len(ratings1)):

                        #check if either rating is a nan
                        if np.isnan(ratings1.iloc[index2]) or np.isnan(ratings2.iloc[index2]):
                            nans += 1
                            continue

                        #calculate weighted manhattan distance to other rows
                        distance_sum += abs(ratings1.iloc[index2] - ratings2.iloc[index2])

                        #euclidean
                        # distance_sum += (ratings1.iloc[index2] - ratings2.iloc[index2]) ** 2
                    
                    #if they're all nans, set distance to -1 so we can filter them out later
                    if nans == total_ratings:
                        distances.append((user2,-1))
                    else:
                        #calculate weighted distance (perhaps test it without weights too?)
                        distance = (total_ratings/(total_ratings - nans)) * distance_sum

                        #euclidean
                        # distance = math.sqrt((total_ratings/(total_ratings - nans)) * distance_sum)

                        #store distance along with its user
                        distances.append((user2,distance))
            
                #find the top k users with the smallest distance, ignore -1s
                similar_users = []
                distances = sorted(distances,key=lambda x: x[1])
                
                if k > len(distances):
                    k = len(distances)
                
                #only focus on valid distances
                for x in range(k):
                    if distances[x][1] == -1:
                        continue
                    similar_users.append(distances[x][0])

                #if there aren't any valid, raise an error (this shouldn't happen)
                if len(distances) == 0:
                    raise Exception(f'ERROR, distances is empty: {distances}')

                #in the rare occasion that all distances are -1, pick a user to impute
                if len(similar_users) == 0:
                    similar_users.append(distances[0][0])

                #impute that column with the average
                column_average = [int(matrix.loc[x].iloc[index]) for x in similar_users]

                #or impute it with 0 if we didn't reach enough k
                # column_average = [int(matrix.loc[x].iloc[index]) for x in similar_users] if len(similar_users) >= k else 0

                #we use new_matrix instead of the original one it doesn't influence KNN
                new_matrix.loc[user1].iloc[index] = sum(column_average)/len(similar_users)
            bar.text = f'\t{c.g}-- Processing {c.p}{user1}{c.g}...'
            bar()

    #reset print color
    print(f'{c.E}')

    #set row index as column user_id
    new_matrix['user_id'] = new_matrix.index
    return new_matrix
        

def main(user=None):
    
    #Get and process data
    user_data, anime_data, ratings_pivot = get_data(user)

    if user:
        print(single_knn(ratings_pivot,2,user).drop('user_id').sort_values(ascending=False).head(20))
        sys.exit()

    #Test RMSE for different amounts of k
    for knn_k in range(1,10):

        #KNN IMPUTE
        test_preds = full_knn(ratings_pivot,knn_k)

        #RMSE
        #of course RMSE is 0 cause it's perfect
        #we need to see if the imputed animes are accurate for new data
        rmse = get_overall_rmse(test_preds,user_data)
        print(f'{c.b}Overall RMSE: {c.g}{c.B}{rmse}{c.E}')


    #Top 5 recommendations
    # if user:
        #get top 5 top-rated movies the user hasn't watched
        #print(get_user_recommendations(user,5,preds,anime_data))
        # print(get_user_RMSE(user)[0])

if __name__ == '__main__':
    #If they input a username, add it
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
