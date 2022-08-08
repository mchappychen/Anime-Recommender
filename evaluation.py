#This file is used to evaluate all algorithms

import ast, time, sys, math
from urllib.parse import non_hierarchical
from alive_progress import alive_bar
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from matrix_factorization_SVD import normalize, SVD
from sklearn.preprocessing import MultiLabelBinarizer
from TFIDF import cosine_similarity, remove_bad_plots

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

#returns user_data,anime_data,ratings_pivot, predictions
def get_data():
    print(f'{c.b}Getting users from {c.v}users.csv.gz{c.b} and all animes from {c.v}animes.csv.gz{c.b}...{c.E}')
    start_time = time.time()

    user_data = pd.read_csv('users.csv.gz').head(2000).drop('Unnamed: 0',axis=1)
    anime_data = pd.read_csv('animes.csv.gz').drop('Unnamed: 0',axis=1)

    new_ratings = ast.literal_eval(open('test_ratings.txt').read())

    #extract ratings from each user
    print(f'{c.b}Extracting user ratings...{c.g}')
    ratings = []
    with alive_bar(len(user_data),bar='bubbles',spinner=None,dual_line=True) as bar:
        for _,row in user_data.iterrows():
            bar()
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
    
    return anime_data,ratings_pivot,new_ratings

def eval_SVD(ratings_pivot,new_ratings):
    #SVD first, impute by row-average, subtract mean, do SVD, add mean, then make prediction
    print(f'{c.b}Imputing...{c.E}')
    start_time = time.time()
    ratings_imputed = ratings_pivot.where(ratings_pivot.notna(), ratings_pivot.mean(axis=1), axis=0)
    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}{c.b}s. Normalizing...{c.E}')
    start_time = time.time()
    ratings_normalized,ratings_mean = normalize(ratings_imputed)
    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}{c.b}s. Factorizing...{c.E}')
    start_time = time.time()
    U_train,sigma_train,Vt_train = SVD(ratings_normalized,25)
    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}{c.b}s. Making predictions...{c.E}')
    preds = pd.DataFrame(np.dot(np.dot(U_train, sigma_train), Vt_train) + ratings_mean.reshape(-1, 1), columns = ratings_pivot.columns)
    preds.index = ratings_pivot.index
    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}{c.b}s{c.E}')

    #calculate RMSE
    svd_rmse = 0
    N = 0
    with alive_bar(len(new_ratings),bar='bubbles',spinner=None,dual_line=True) as bar:
        for user in new_ratings:
            bar()
            ratings_dict = new_ratings[user]
            for anime_id in ratings_dict:
                #our prediction matrix may not have the anime
                if anime_id not in preds.columns:
                    continue
                #it may not even have the new user
                if user not in preds.index:
                    continue
                svd_rmse += (preds.loc[user,anime_id] - ratings_dict[anime_id])**2
                N += 1
    svd_rmse = math.sqrt(svd_rmse / N)
    print(svd_rmse)

def eval_KNN(matrix,new_ratings,k):

    #We're gonna let k = 5
    k = 5

    knn_1_rmse = 0 
    N_1 = 0
    knn_2_rmse = 0
    N_2 = 0
    knn_3_rmse = 0
    N_3 = 0
    knn_4_rmse = 0
    N_4 = 0
    knn_5_rmse = 0
    N_5 = 0

    print(f'{c.g}')
    
    with alive_bar(len(new_ratings),bar='bubbles',spinner=None,dual_line=True) as bar:
        #for each user in new_ratings
        for user1 in new_ratings:
            bar.text = f'Predicting for {user1}...'
            bar()

            #make sure the user exists in matrix
            if user1 not in matrix.index:
                continue

            #get the row that matches this user
            user_row = matrix.loc[user1]

            #for each anime in new_ratings, predict it
            for anime1 in new_ratings[user1]:
                
                #we can't predict on animes that no one ever rated
                if anime1 not in matrix.columns:
                    continue

                #calculate distance for this user to all other users
                distances = []

                #calculate the distance to every other row
                for user2,matrix_row in matrix.iterrows():
                    
                    #make sure we're not repeating our own row
                    if user1 == user2:
                        continue
                    
                    #make sure this user rated this anime before
                    if np.isnan(matrix.loc[user2,anime1]):
                        continue

                    #keep track of how many nans there are
                    nans = 0
                    total_ratings = len(matrix_row)
                    distance_sum = 0
                    
                    #go through each rating from left to right
                    for index in range(len(matrix_row)):

                        #check if rating is a nan
                        if np.isnan(matrix_row.iloc[index]) or np.isnan(user_row.iloc[index]):
                            nans += 1
                            continue

                        #calculate weighted manhattan distance to other rows
                        distance_sum += abs(matrix_row.iloc[index] - user_row.iloc[index])
                    
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
                x = 0
                while len(similar_users) < k and x < len(distances):
                    if distances[x][1] == -1:
                        x+=1
                        continue
                    similar_users.append(distances[x][0])
                    x+=1
                del x

                #if there aren't any valid, raise an error (this shouldn't happen)
                if len(distances) == 0:
                    raise Exception(f'ERROR, distances is empty: {distances}')

                #in the rare occasion that all distances are -1, pick a user to impute
                if len(similar_users) == 0:
                    similar_users.append(distances[0][0])

                column_average_5 = [int(matrix.loc[x,anime1]) for x in similar_users]
                predicted_rating_5 = sum(column_average_5)/len(similar_users)
                if len(similar_users) == 5:
                    similar_users.pop()
                column_average_4 = [int(matrix.loc[x,anime1]) for x in similar_users]
                predicted_rating_4 = sum(column_average_4)/len(similar_users)
                if len(similar_users) == 4:
                    similar_users.pop()
                column_average_3 = [int(matrix.loc[x,anime1]) for x in similar_users]
                predicted_rating_3 = sum(column_average_3)/len(similar_users)
                if len(similar_users) == 3:
                    similar_users.pop()
                column_average_2 = [int(matrix.loc[x,anime1]) for x in similar_users]
                predicted_rating_2 = sum(column_average_2)/len(similar_users)
                if len(similar_users) == 2:
                    similar_users.pop()
                column_average_1 = [int(matrix.loc[x,anime1]) for x in similar_users]
                predicted_rating_1 = sum(column_average_1)/len(similar_users)

                #calculate RMSE
                knn_1_rmse += (predicted_rating_1 - new_ratings[user1][anime1])**2
                N_1 += 1
                knn_2_rmse += (predicted_rating_2 - new_ratings[user1][anime1])**2
                N_2 += 1
                knn_3_rmse += (predicted_rating_3 - new_ratings[user1][anime1])**2
                N_3 += 1
                knn_4_rmse += (predicted_rating_4 - new_ratings[user1][anime1])**2
                N_4 += 1
                knn_5_rmse += (predicted_rating_5 - new_ratings[user1][anime1])**2
                N_5 += 1

    knn_1_rmse = math.sqrt(knn_1_rmse / N_1)
    knn_2_rmse = math.sqrt(knn_2_rmse / N_2)
    knn_3_rmse = math.sqrt(knn_3_rmse / N_3)
    knn_4_rmse = math.sqrt(knn_4_rmse / N_4)
    knn_5_rmse = math.sqrt(knn_5_rmse / N_5)

    print(f'{c.b}RMSE for k={c.p}1{c.b}: {c.B}{c.g}{knn_1_rmse}{c.E}')
    print(f'{c.b}RMSE for k={c.p}2{c.b}: {c.B}{c.g}{knn_2_rmse}{c.E}')
    print(f'{c.b}RMSE for k={c.p}3{c.b}: {c.B}{c.g}{knn_3_rmse}{c.E}')
    print(f'{c.b}RMSE for k={c.p}4{c.b}: {c.B}{c.g}{knn_4_rmse}{c.E}')
    print(f'{c.b}RMSE for k={c.p}5{c.b}: {c.B}{c.g}{knn_5_rmse}{c.E}')

def eval_COSINE_JACCARD(anime_data,ratings_pivot,new_ratings):
    
    mlb = MultiLabelBinarizer()

    #replace NA values with string
    anime_data['Rating'].fillna('None', inplace=True)

    #Transform the column values into arrays
    anime_data['Rating'] = anime_data['Rating'].apply(lambda x: ['Rating_' + y for y in x.replace("'",'').strip("[]").split(', ')]  )

    #use MultiLabelBinarizer to encode columns into [1,0,1,0] vectors
    anime_data = anime_data.join(pd.DataFrame(mlb.fit_transform(anime_data['Rating']) , columns=mlb.classes_).drop(['Rating_None'],axis=1) )
    anime_data['Type'].fillna('None', inplace=True)
    anime_data['Type'] = anime_data['Type'].apply(lambda x: ['Type_' + y for y in x.replace("'",'').strip("[]").split(', ')]  )
    anime_data = anime_data.join(pd.DataFrame(mlb.fit_transform(anime_data['Type']) , columns=mlb.classes_).drop(['Type_Unknown'],axis=1) )
    anime_data['Themes'].fillna('None', inplace=True)
    anime_data['Themes'] = anime_data['Themes'].apply(lambda x: ['Theme_' + y for y in x.replace("'",'').strip("[]").split(', ')]  )
    anime_data = anime_data.join(pd.DataFrame(mlb.fit_transform(anime_data['Themes']) , columns=mlb.classes_).drop(['Theme_None'],axis=1))
    anime_data['Genres'].fillna('None', inplace=True)
    anime_data['Genres'] = anime_data['Genres'].apply(lambda x: ['Genre_' + y for y in x.replace("'",'').strip("[]").split(', ')]  )
    anime_data = anime_data.join(pd.DataFrame(mlb.fit_transform(anime_data['Genres']) , columns=mlb.classes_).drop(['Genre_None'],axis=1))

    cosine_rmse = 0
    jaccard_rmse = 0
    N = 0

    print(f'{c.g}')
    #loop through all the new ratings if the user exists in ratings_pivot
    with alive_bar(sum(1 for x in new_ratings for y in new_ratings[x] if x in ratings_pivot.index),bar='bubbles',spinner=None,dual_line=True) as bar:

        #for each user
        for user in new_ratings:

            #skip if the user isn't in ratings matrix
            if user not in ratings_pivot.index:
                # print(f'{c.p} {user}{c.y} not in ratings_pivot. Skipping...{c.E}')
                continue

            bar.text = f'{c.b}Cosine RMSE: {c.g}{round(math.sqrt(cosine_rmse/N),3) if N != 0 else 0}{c.b} Jaccard RMSE: {c.g}{round(math.sqrt(jaccard_rmse/N),3) if N != 0 else 0}{c.b} Predicting...{c.g}'

            #(anime,rating) get the list of animes the user has rated
            original_user_ratings = [(index,value) for index,value in ratings_pivot.loc[user].iteritems() if not np.isnan(value)]
            
            #for each new anime the user has rated
            for new_anime in new_ratings[user]:

                bar()

                #if the anime is brand new, skip it
                if not new_anime in anime_data['_id'].values:
                    print(f'{c.y}New anime {c.p}{new_anime}{c.y} does not exist in my database yet{c.g}')
                    continue

                #get the [0,1,0] vector for new_anime\
                new_anime_vector = list(anime_data[anime_data['_id']==new_anime].drop(['_id','title','rating','Type','Status','Producers','Studios','Genres','Themes','Demographics','Rating','plot'],axis=1).iloc[0])
                
                #anime_id
                most_similar_cosine_anime = None
                biggest_cosine_similarity = -1

                most_similar_jaccard_anime = None
                biggest_jaccard_similarity = -1


                for old_anime in original_user_ratings:

                    #Make sure the animes are different (shouldn't be possible)
                    if new_anime == old_anime[0]:
                        print(f'{c.r}ERROR: new_anime {new_anime} = old_anime {old_anime[0]}{c.E}')
                        sys.exit()

                    #in the rare occassion that an old anime isn't in ratings_pivot (dont know how), skip
                    if old_anime[0] not in anime_data['_id'].values:
                        print(f'{c.y}Old anime {c.p}{old_anime[0]}{c.y} is not in anime_data. Skipping...')
                        continue

                    #get the vector for old_anime
                    try:
                        old_anime_vector = list(anime_data[anime_data['_id']==old_anime[0]].drop(['_id','title','rating','Type','Status','Producers','Studios','Genres','Themes','Demographics','Rating','plot'],axis=1).iloc[0])
                    except Exception as e:
                        print(e)
                        print(new_anime,old_anime,old_anime[0] in anime_data['_id'].values)
                        sys.exit()

                    #if one of the vectors is all 0, we can't calculate similarity
                    if not any(old_anime_vector) or not any(new_anime_vector):
                        continue

                    #calculate cosine_similarity
                    cosine_sim = sum(old_anime_vector[x]*new_anime_vector[x] for x in range(len(old_anime_vector))) / (math.sqrt(sum(x for x in old_anime_vector if x != 0)) * math.sqrt(sum(x for x in new_anime_vector if x != 0)))
                    
                    #find the highest similarity
                    if cosine_sim > biggest_cosine_similarity:
                        most_similar_cosine_anime = old_anime[0]
                        biggest_cosine_similarity = cosine_sim
                    
                    #calculate jaccard similarity
                    jaccard_sim = sum(1 for x in range(len(old_anime_vector)) if old_anime_vector[x] == new_anime_vector[x] and old_anime_vector[x] != 0) / sum(1 for x in range(len(old_anime_vector)) if old_anime_vector[x] == 1 or new_anime_vector[x] == 1)
                    if jaccard_sim > biggest_jaccard_similarity:
                        most_similar_jaccard_anime = old_anime[0]
                        biggest_jaccard_similarity = jaccard_sim
                
                #In the impossible case that there's no similar animes
                if most_similar_cosine_anime == None or most_similar_jaccard_anime == None:
                    print(f'{c.r}No similar animes for {new_anime} for {c.p}{user}{c.E}')
                    sys.exit()

                #get the rating for the most similar anime the user rated
                most_similar_cosine_anime_rating = [x[1] for x in original_user_ratings if x[0] == most_similar_cosine_anime][0]

                #calculate RMSE for cosine
                cosine_rmse += (most_similar_cosine_anime_rating - new_ratings[user][new_anime])**2

                most_similar_jaccard_anime_rating = [x[1] for x in original_user_ratings if x[0] == most_similar_jaccard_anime][0]
                jaccard_rmse += (most_similar_jaccard_anime_rating - new_ratings[user][new_anime])**2
                N += 1

    
    cosine_rmse = math.sqrt(cosine_rmse/N)
    print(f'{c.b}Cosine RMSE: {c.g}{cosine_rmse}{c.E}')
    jaccard_rmse = math.sqrt(jaccard_rmse/N)
    print(f'{c.b}Jaccard RMSE: {c.g}{jaccard_rmse}{c.E}')

def eval_TFIDF(anime_data,ratings_pivot,new_ratings):
    
    anime_data = remove_bad_plots(anime_data)

    stop_words = 'the,to,and,of,a,what,through,in,which,when,is,by,with,so,for,that,or,on,about,there,where,as,has,from,their,who,this,an,synopsis,are,they,was,been,be,at,but,into,will,our,it,have,'.split(',')
    white_list = set('qwertyuiopasdfghjklzxcvbnm ')
    
    #keep track of how many plots each word occurs in
    words = dict()

    #makes a plot into a dictionary of word counts
    def make_dict(plot):
        #convert plot into a list of words
        plot_list = ''.join(filter(white_list.__contains__,plot.lower())).replace('-', ' ').split(' ')

        #remove the stop words
        # plot_list = [word for word in plot_list if word not in stop_words]

        #add the plot_list to words without repeats so they can be used to calculate IDF
        for word in set(plot_list):
            words[word] = words.get(word,0)+1

        #make a dictionary of {word:TF}
        plot_dict = dict()
        for word in plot_list:
            plot_dict[word] = plot_dict.get(word,0)+1
        for x in plot_dict:
            plot_dict[x] = plot_dict[x]/len(plot_list)

        return plot_dict

    anime_data['tf'] = anime_data.apply(lambda row: make_dict(row['plot']),axis=1)

    #Calculate the IDF for each word, IDF = log(total animes / # animes the word is in)
    idf = dict()
    #word is (word,count)
    for word in words.items():
        idf[word[0]] = round(math.log(len(anime_data)/word[1]),6)

    #returns a dict of TFIDF given a dict of TF
    def get_tfidf(tf):
        tfidf = dict()
        for x in tf:
            tfidf[x] = tf[x] * idf[x]
        return tfidf

    #multiply each TF by IDF
    anime_data['tfidf'] = anime_data.apply(lambda row: get_tfidf(row['tf']),axis=1)

    def recommend(anime_id,animes):
        #check if anime_id exists in animes:
        if not any(animes['_id']==anime_id):
            print(animes[animes['_id']==anime_id])
            print(f'{c.r}Error: Your anime_id: {c.y}{anime_id}{c.r} does not exist or does not have a plot{c.E}')
            sys.exit()

        #Get the tfidf for anime_id
        anime_id_tfidf = animes.loc[animes['_id']==anime_id]['tfidf'].iloc[0]

        #Calculate the similarities for each anime
        animes['cosine_similarity'] = animes.apply(lambda row: cosine_similarity( row['tfidf'] , anime_id_tfidf ), axis=1)

        #Get the top 10 best similarities
        return animes.sort_values('cosine_similarity',ascending=False).head(10).drop(['rating','Status','Producers','Studios','Genres','Themes','Type','Demographics','tf','tfidf','Rating'],axis=1)[1:]
    
    tfidf_rmse = 0
    N = 0

    with alive_bar(sum(1 for x in new_ratings for y in new_ratings[x] if x in ratings_pivot.index),bar='bubbles',spinner=None,dual_line=True) as bar:

        #for each user in test set
        for user in new_ratings:

            #make sure this user is selected
            if user not in ratings_pivot.index:
                continue

            original_user_ratings = [(index,value) for index,value in ratings_pivot.loc[user].iteritems() if not np.isnan(value)]

            #for each new rating the user made
            for new_anime in new_ratings[user]:
                
                bar()
            
                #make sure the anime exists in our database
                if new_anime not in anime_data['_id'].values:
                    # print(f'{c.y}New anime {new_anime} is not in anime_data{c.E}')
                    continue
                
                most_similar_anime = None
                cos_similarity = 0

                #for each old rating the user made
                #(anime_id, rating)
                for old_anime in original_user_ratings:

                    #make sure this anime exists in ratings_pivot
                    #A lot of anime won't since their ratings are bad
                    if old_anime[0] not in anime_data['_id'].values:
                        # print(f'{c.y}Old anime {old_anime[0]} is not in anime_data')
                        continue

                    #find the cosine similarity between the TFIDFs for the 2 animes
                    sim = cosine_similarity(anime_data[anime_data['_id']==old_anime[0]]['tfidf'].iloc[0],anime_data[anime_data['_id']==new_anime]['tfidf'].iloc[0])
                    
                    if sim > cos_similarity:
                        cos_similarity = sim
                        most_similar_anime = old_anime
                
                if most_similar_anime == None:
                    print(f'{c.r}Error: most_simiar_anime is None for {user} This user doesnt have a valid anime history with good plot{c.E}')
                    continue

                #find the rating of the most similar anime the user gave
                #calculate RMSE
                tfidf_rmse += (most_similar_anime[1] - new_ratings[user][new_anime])**2
                N+=1
                bar.text = f'TFIDF RMSE: {round(math.sqrt(tfidf_rmse/N),3) if not N==0 else 0}'

    tfidf_rmse = math.sqrt(tfidf_rmse/N)
    print(f'{c.b}TFIDF RMSE: {c.g}{round(tfidf_rmse,3)}{c.E}')





def main():
    #get the files
    anime_data,ratings_pivot,new_ratings = get_data()

    #eval SVD (1.5 RMSE)
    eval_SVD(ratings_pivot,new_ratings)

    #KNN takes too long, we'll set users=600 and run it overnight (1.7 RMSE)
    eval_KNN(ratings_pivot,new_ratings,5)

    #Cosine and Jaccard also takes too long due to my algorithm
    eval_COSINE_JACCARD(anime_data,ratings_pivot,new_ratings)

    eval_TFIDF(anime_data,ratings_pivot,new_ratings)


if __name__ == '__main__':
    main()
