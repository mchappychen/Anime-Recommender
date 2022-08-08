#This file is used to find similar animes using the Jaccard/Cosine metrics


from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import sys, ast, time, math

pd.set_option('display.max_columns',None)

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

#returns animes
def get_data(user=None):
    anime_data = pd.read_csv('animes.csv.gz').drop('Unnamed: 0',axis=1)

    return anime_data

#finds the top 10 similar animes to this anime based on jaccard similarity given anime_id and anime_Data
def jaccard_similarity(anime_id,anime_data):

    print(f'{c.b}Calculating Jaccard Similarity for anime_id: {c.p}{anime_id}{c.b}...{c.E}')
    start_time = time.time()

    anime1_features = set()

    #find the features of anime1 by anime_id
    anime1 = anime_data[anime_data['_id']==anime_id]

    #put anime1 features into a set
    for column_name,column_data in anime1.iteritems():
        #column_data is series
        column_data = column_data.values
        if column_name == 'Rating':
            anime1_features.add(column_data[0])
            continue
        if column_name in ['Genres','Themes','Demographics','Producers','Studios']:
            #skip NaN
            if pd.isna(column_data[0]):
                continue
            if '[' in column_data[0]:
                for x in ast.literal_eval(column_data[0]):
                    anime1_features.add(x)
            else:
                anime1_features.add(column_data[0])

    anime_data['jaccard'] = np.nan

    #calculate jaccard for each row
    for _,anime2 in anime_data.iterrows():

        anime2_features = set()

        for column_name,column_data in anime2.iteritems():
            if column_name == 'Rating':
                anime2_features.add(column_data)
            if column_name in ['Genres','Themes','Demographics','Producers','Studios']:
                if pd.isna(column_data):
                    continue
                if '[' in column_data:
                    for x in ast.literal_eval(column_data):
                        anime2_features.add(x)
                else:
                    anime2_features.add(column_data)

        #jaccard = intersection / union
        jaccard_sim = len(anime1_features & anime2_features)/len(anime1_features|anime2_features)
        anime_data.loc[anime_data['_id']==anime2['_id'],'jaccard'] = jaccard_sim
  
    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}s{c.E}')
    return anime_data.sort_values('jaccard',ascending=False).iloc[1:].head(10)

def cosine_similarity(anime_id,anime_data):

    print(f'{c.b}Calculating cosine similarity for anime_id: {c.p}{anime_id}{c.b}...{c.E}')
    start_time = time.time()

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
    
    #make empty column for cosine similarity
    cosine_similarity_column = []

    anime1 = anime_data[anime_data['_id']==anime_id].drop(['_id','title','rating','Type','Status','Producers','Studios','Genres','Themes','Demographics','Rating','plot'],axis=1)
    anime1_vector = [x.values[0] for _,x in anime1.iteritems()]

    anime_data_encoded = anime_data.drop(['_id','title','rating','Type','Status','Producers','Studios','Genres','Themes','Demographics','Rating','plot'],axis=1)
    for _,anime2 in anime_data_encoded.iterrows():
        anime2_vector = [x for _,x in anime2.iteritems()]

        #cosine similarity = dot product / (length a x length b)
        #dot product = however many same 1s there are
        #length a = sqrt(number of 1s in a)

        #if one of the vectors is all 0, we can't compute cosine similarity, so just set it to 0
        if (math.sqrt(sum(x for x in anime1_vector if x != 0)) * math.sqrt(sum(x for x in anime2_vector if x != 0))) == 0:
            cosine_sim = 0
        else:
            #calculate cosine similarity based on above formula
            cosine_sim = sum(anime1_vector[x]*anime2_vector[x] for x in range(len(anime1_vector))) / (math.sqrt(sum(x for x in anime1_vector if x != 0)) * math.sqrt(sum(x for x in anime2_vector if x != 0)))
 
        cosine_similarity_column.append(cosine_sim)

    #set the cosine similarity column
    anime_data['cosine'] = cosine_similarity_column

    anime_data = anime_data[['_id','title','cosine','rating','Type','Status','Producers','Studios','Genres','Themes','Demographics','Rating','plot']]

    print(f'{c.b}Done. Time took: {c.p}{round(time.time()-start_time)}s{c.E}')

    return anime_data.sort_values('cosine',ascending=False).iloc[1:].head(10)


def main(anime_id=None):
    anime_data = get_data()

    print(jaccard_similarity(anime_id,anime_data)[['title','_id','jaccard']])

    print(cosine_similarity(anime_id,anime_data)[['title','_id','cosine']])

if __name__ == '__main__':
    #If they input an animeID, add it
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
