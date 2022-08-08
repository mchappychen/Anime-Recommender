#This file re-scrapes the users to get an updated ratings list

## CONFIG
batch_size = 100
wait_time = 100 #in seconds
access_token = 'GET YOUR OWN MAL ACCESS TOKEN'
##

from pymongo import MongoClient
from requests_html import AsyncHTMLSession
import json, time, sys, requests, os, asyncio
import pandas as pd

asession = AsyncHTMLSession()

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

users = pd.read_csv('users.csv.gz')['_id']

#connect to my database in mongodb
print(f'{c.b}Connecting to database...{c.E}')
client = MongoClient("GET YOUR OWN MONGODB ADDRESS",tlsAllowInvalidCertificates=True)
print(f'{c.g}Connected!{c.E}')
database = client['MyAnimeList']

last_entry = database['Misc'].find({},{'_id':0})[0]['last_entry']
print(f'{c.b}Last user index scraped: {c.B}{c.p}{last_entry}{c.E}')

#scraped data
batch = []

#this will be True with Anti-DoS is triggered
wait = False

async def scrape(user):

    global wait

    #visit website
    response = await asession.get(
        f'https://api.myanimelist.net/v2/users/{user}/animelist?limit=700&fields=list_status&nsfw=true', 
        headers = {'Authorization':f'Bearer {access_token}'}
    )

    if response.status_code != 200:
        try:
            if json.loads(response.text)['error'] == 'not_permitted':
                print(f' {c.y}User {c.p}{c.B}{user}{c.E}{c.y} does not allow us to scrape them. Skipping...{c.E}')
            elif json.loads(response.text)['error'] == 'not_found':
                print(f' {c.y}User {c.p}{c.B}{user}{c.E}{c.y} has deleted their profile. Skipping...{c.E}')
            else:
                print(f' {c.y}{response.status_code} status code for {user}: {response.text}{c.E}')
        except Exception as e:
            print(f'{c.r}ERROR Anti-Dos Triggered!{c.E}')
            wait = True
        return
    
    try:
        data = json.loads(response.text)['data']
    except Exception as e:
        print(f"  {c.r}ERROR for {user}: {json.loads(response.text)['message']}, {json.loads(response.text)['error']}{c.E}")
        return

    ratings = []
    for rating in data:
        try:
            ratings.append({
                'animeID' : rating['node']['id'],
                'rating' : rating['list_status']['score'],
                'status' : rating['list_status']['status']
            })
        except Exception as e:
            print(f'{c.r} ERROR for {user}: {e}')

    #add the ratings to batch
    batch.append({'_id':user,'ratings':ratings})


while last_entry < len(users):
    start_time = time.time()

    #scrape the users
    asession.run(*[lambda x=y: scrape(x) for y in users[last_entry:last_entry+batch_size]])

    #increase wait_time and decrease batch_size if we get Anti-DoS
    if wait:
        wait_time += 5
        batch_size -= 5
        wait = False
        batch.clear()
        print(f'{c.b}Waiting {c.p}{c.B}{wait_time+60}{c.E}{c.b} seconds...{c.E}')
        time.sleep(wait_time+60)
        print(f'{c.b}Rescraping...{c.E}')
        continue


    #check if we've scraped any ratings
    if len(batch):
        print(f'\U00002705 {c.b}Finished scraping. {c.g}{c.B}+{len(batch)}{c.E}{c.b} users\' ratings. Average Speed: {c.v}{c.B}{round(len(batch)/(time.time()-start_time),1)}{c.b} per second.{c.E}')
        #insert into database
        database['Users'].insert_many(batch)
    else:
        #if none were scraped, continue on
        print(f'{c.y}No ratings were scraped for this batch.{c.E}')

    batch.clear()
    
    #update last_entry
    last_entry += batch_size
    database['Misc'].update_one({},{ "$set": { "last_entry": last_entry} })

    print(f'{c.b}Waiting {c.p}{c.B}{wait_time}{c.E}{c.b} seconds...{c.E}')
    time.sleep(wait_time)




