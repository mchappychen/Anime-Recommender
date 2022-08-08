#This file uses TFIDF to find animes with similar plot synopsis

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import sys, ast, time, math, re, json
from operator import itemgetter


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

#gets the data
def get_data():
    return pd.read_csv('animes.csv.gz').drop('Unnamed: 0',axis=1)

#calculates cosine similarity between 2 dicts
def cosine_similarity(dict_A,dict_B):
    if dict_A == dict() or dict_B == dict():
        return 0
    #A dot B / ||A|| x ||B||
    return sum(dict_A[x]*dict_B[y] for x in dict_A for y in dict_B if x==y) / (math.sqrt(sum(x**2 for x in dict_A.values())) *  math.sqrt(sum(x**2 for x in dict_B.values())))


#Builds the an IDF table, so it's easier to calculate the TF-IDF later on
def process_animes(animes):

    print(f'{c.b}Calculating TF-IDF...{c.E}')

    #remove all stop words to make computation easier, stop words were gathered by selecting out of top 50 most-common words
    stop_words = 'the,to,and,of,a,what,through,in,which,when,is,by,with,so,for,that,or,on,about,there,where,as,has,from,their,who,this,an,synopsis,are,they,was,been,be,at,but,into,will,our,it,have,'.split(',')
    
    #only keep letters
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

    animes['tf'] = animes.apply(lambda row: make_dict(row['plot']),axis=1)

    # This prints out the most common words
    # print(sorted(words.items(), key=lambda item: item[1],reverse=True)[60:90])

    #Calculate the IDF for each word, IDF = log(total animes / # animes the word is in)
    idf = dict()
    #word is (word,count)
    for word in words.items():
        idf[word[0]] = round(math.log(len(animes)/word[1]),6)

    #returns a dict of TFIDF given a dict of TF
    def get_tfidf(tf):
        tfidf = dict()
        for x in tf:
            tfidf[x] = tf[x] * idf[x]
        return tfidf

    #multiply each TF by IDF
    animes['tfidf'] = animes.apply(lambda row: get_tfidf(row['tf']),axis=1)

    print(f'{c.b}Done calculating TF-IDF. There are {c.p}{len(words)}{c.b} unique words and {c.p}{len(animes)}{c.b} animes with valid plots.')

    return animes

#recommends animes based on cosine_similarity of TF-IDF 
def recommend(anime_id,animes):

    #check if anime_id exists in animes:
    if not any(animes['_id']==anime_id):
        print(animes[animes['_id']==anime_id])
        print(f'{c.r}Error: Your anime_id: {c.y}{anime_id}{c.r} does not exist or does not have a plot{c.E}')
        return

    print(f'{c.b}Finding recommendations for anime_id: {c.B}{c.p}{anime_id}{c.E}{c.b} ({c.B}{c.p}{animes.loc[animes["_id"]==anime_id]["title"].iloc[0]}{c.E}{c.b}) ...')

    #Get the tfidf for anime_id
    anime_id_tfidf = animes.loc[animes['_id']==anime_id]['tfidf'].iloc[0]

    #Calculate the similarities for each anime
    animes['cosine_similarity'] = animes.apply(lambda row: cosine_similarity( row['tfidf'] , anime_id_tfidf ), axis=1)

    #Set the color to green
    print(f'{c.g}')

    #Get the top 10 best similarities
    return animes.sort_values('cosine_similarity',ascending=False).head(10).drop(['rating','Status','Producers','Studios','Genres','Themes','Type','Demographics','tf','tfidf','Rating'],axis=1)[1:]


#returns animes with real plots
def remove_bad_plots(anime_data):

    #remove animes where plots are NA
    possible_recommendations = anime_data[~anime_data['plot'].isna()]

    #remove animes where plot is "add a plot"
    possible_recommendations = possible_recommendations[~(possible_recommendations['plot'] == 'No synopsis information has been added to this title. Help improve our database by adding a synopsis here.')]
    possible_recommendations = possible_recommendations[~(possible_recommendations['plot'] == 'No synopsis has been added for this series yet.Click here to update this information.')]

    return possible_recommendations

def main():
    #get anime data
    anime_data = get_data()

    #remove animes without a plot 
    possible_recommendations = remove_bad_plots(anime_data)
    
    #Calculate TF-IDF for anime
    recommendations = process_animes(possible_recommendations)


    while True:
        anime_id = input(f'{c.B}{c.b}Input an anime_id: {c.v}')
        if not anime_id:
            print(f'{c.E}')
            break
        #Get top 10 similar animes
        print(recommend(int(anime_id),possible_recommendations))
        print(f'{c.E}')


if __name__ == '__main__':
    main()
