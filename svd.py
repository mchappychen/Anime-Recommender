#This file performs cross validation on SVD

### CONFIG
# svd_k = 2 #number of top features to keep in SVD
# knn_k = 1 #number of neighbors for KNN Imputer
user_rows = 10000 #number of users to get. set to None for all
### END CONFIG

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
from sklearn.model_selection import KFold
import json, time, sys, requests, os

# pd.set_option('display.max_columns',None)

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
    for index,row in user_data.iterrows():
        for anime in ast.literal_eval(row['ratings']):
            #skip the unrated animes
            if anime['rating'] == 0:
                continue
            ratings.append({
                'user_id':row['_id'],
                'anime_id':anime['animeID'],
                'rating':anime['rating']})


    #Format so columns are animes, and rows are users                                                                                                                                                                                                                
    ratings_pivot = pd.DataFrame(ratings).pivot(index = 'user_id', columns ='anime_id', values = 'rating')

    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}s{c.b}. Sample list of users: {c.v}{user_data.sample(3)["_id"].tolist()}{c.E}')
    
    return user_data,anime_data,ratings_pivot

#returns U,sigma,Vt from Singular Value Decomposition
def SVD(ratings,svd_k):
    print(f'{c.b}Doing SVD for top {c.p}{svd_k}{c.b} features...{c.E}')
    U, sigma, Vt = svds(ratings, k = svd_k)
    sigma = np.diag(sigma)
    # print(ratings.shape,U.shape,sigma.shape,Vt.shape)
    return U,sigma,Vt

#normalize each row by subtracting from their row avg and return their means
def normalize(matrix):
    means = np.mean(matrix.values,axis=1)
    return matrix.values - means.reshape(-1,1),means

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

def main(user=None):
    
    #Get and process data
    user_data, anime_data, ratings_pivot = get_data()

    #(step_size,average RMSE)
    training_rmse_list = []

    #split ratings_pivot into train and test
    train,test = train_test_split(ratings_pivot,test_size=0.3)

    #Cross-validate
    for svd_k in range(100,1000,50):
        
        rmse_list = []

        #cross-validation 5-fold
        kf = KFold(n_splits = 5, shuffle = True)
        
        print(f'{c.b}Doing cross-validation for svd_k = {c.p}{svd_k}{c.E}')
        start_time = time.time()

        for train_index,validate_index in kf.split(train):

            #split the training into train and validate sets
            train_ratings = train.iloc[train_index]
            validate_ratings = train.iloc[validate_index]

            #impute by row average
            train_ratings_imputed = train_ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
            validate_ratings_imputed = validate_ratings.apply(lambda row: row.fillna(row.mean()), axis=1)

            #subtract their means (get rid of user bias in ratings)
            train_ratings_normalized,train_user_ratings_mean = normalize(train_ratings_imputed)
            validate_ratings_normalized,validate_user_ratings_mean = normalize(validate_ratings_imputed)

            #train SVD weights on train-set
            U_train,sigma_train,Vt_train = SVD(train_ratings_normalized,svd_k)
            U_validate,sigma_validate,Vt_validate = SVD(validate_ratings_normalized,svd_k)

            #predict test_data from train sigma and Vt, add the means back
            validate_preds = pd.DataFrame(np.dot(np.dot(U_validate, sigma_train), Vt_train) + validate_user_ratings_mean.reshape(-1, 1), columns = validate_ratings.columns)

            #add the user_ids back (from index after pivot)
            validate_preds['user_id'] = validate_ratings.index

            #RMSE
            rmse = get_overall_rmse(validate_preds,user_data)
            rmse_list.append(rmse)
        
        average_rmse = sum(rmse_list)/len(rmse_list)
        training_rmse_list.append((svd_k,average_rmse))

        print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}s{c.b} svd_k of {c.p}{svd_k}{c.b} has average CV-RMSE of {c.g}{round(average_rmse,2)}{c.E}')

        #Compare the CV RMSE to test RMSE
        train_ratings_imputed = train.apply(lambda row: row.fillna(row.mean()), axis=1)
        test_ratings_imputed = test.apply(lambda row: row.fillna(row.mean()), axis=1)
        
        train_ratings_normalized,train_user_ratings_mean = normalize(train_ratings_imputed)
        test_ratings_normalized,test_user_ratings_mean = normalize(test_ratings_imputed)

        U_train,sigma_train,Vt_train = SVD(train_ratings_normalized,svd_k)
        U_test,sigma_test,Vt_test = SVD(test_ratings_normalized,svd_k)

        test_preds = pd.DataFrame(np.dot(np.dot(U_test, sigma_train), Vt_train) + test_user_ratings_mean.reshape(-1, 1), columns = test.columns)

        test_preds['user_id'] = test.index

        rmse = get_overall_rmse(test_preds,user_data)
        
        print(f'{c.b}svd_k of {c.p}{svd_k}{c.b} has test-RMSE of {c.g}{round(rmse,2)}{c.E}')


    #Top 5 recommendations
    # if user:
    #     print(get_user_recommendations(user,5,preds,anime_data))
        # print(get_user_RMSE(user)[0])

if __name__ == '__main__':
    #If they input a username, add it
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
