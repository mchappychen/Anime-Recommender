#This file scrapes user rating data from MAL

### CONFIG
wait_time = 1 #number of seconds to wait between scraping (avoid Anti-DoS)
access_token = 'GET YOUR OWN ACCESS TOKEN FROM MAL'
limit = 500 #maximum number of ratings to get from a user
min_animes_watched = 10
## END CONFIG

from pymongo import MongoClient
import pymongo
from requests_html import HTMLSession
import json, time, sys, requests, os, asyncio
import pandas as pd

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


#connect to my database in mongodb
print(f'{c.b}Connecting to database...{c.E}')
client = MongoClient("GET YOUR OWN MONGODB ADDRESS",tlsAllowInvalidCertificates=True)
print(f'{c.g}Connected!{c.E}')
database = client['MyAnimeList']

#used to visit webpage
session = HTMLSession()

#used to store groups of data for inserting later (more efficient)
batch = []

#this will be True when Anti-DoS is triggered
wait = False

#keep track of which users we've already scraped. Set has O(1) search time
print(f'{c.b}Getting users we\'ve scraped before...{c.E}')
scraped_users = set(x['_id'] for x in database['Users'].find({},{'_id':1}))
print(f'{c.g}Got the users, there are {c.p}{c.B}{len(scraped_users)}{c.E}{c.g} of them{c.E}')

#asynchrounous so it can be run multiple times at once (concurrent)
async def scrape(user):

    global wait

    #skip users we've scraped before
    scraped_users.add(user)

    # -- This might be the reason why my asynchrounous calling isn't working
    #Get data from API
    response = requests.get(
        f'https://api.myanimelist.net/v2/users/{user}/animelist?limit={limit}&fields=list_status&nsfw=true', 
        headers = {'Authorization':f'Bearer {access_token}'}
    )

    #If we get an error (probably anti-DoS triggered), analyze
    if response.status_code != 200:
        try:
            if json.loads(response.text)['error'] == 'not_permitted':
                print(f' {c.y}User {c.p}{c.B}{user}{c.E}{c.y} does not allow us to scrape them. Skipping...{c.E}')
            else:
                print(f' {c.y}{response.status_code} status code for {user}: {response.text}{c.E}')
        except Exception as e:
            print(f'{c.r}ERROR Anti-Dos Triggered!{c.E}')
            wait = True
        return

    try:
        data = json.loads(response.text)['data']
    except Exception as e:
        print(f"  {c.r}ERROR for {user}: {json.loads(x.text)['message']}, {json.loads(x.text)['error']}{c.E}")
        return
    
    #if this user didn't watch enough animes, skip
    if len(data) < min_animes_watched:
        print(f' {c.y}User {c.p}{c.B}{user}{c.E}{c.y} did not watch enough animes. Skipping...{c.E}')
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
            print(f'{c.r}ERROR for {user}: {e}')

    #add the ratings to batch
    batch.append({'_id':user,'ratings':ratings})


async def main():

    #Go to /users.php to get new users
    print(f'{c.b}Getting new list of users from /users.php ...{c.E}')
    site = session.get('https://myanimelist.net/users.php')

    #Scrape the site for users and only keep the users we haven't scraped before
    new_users = [user_element.text for user_element in site.html.find('td div[style] a[href*="profile"]') if user_element.text not in scraped_users]
    print(f'{c.b}Done. Scraping {c.v}{c.B}{len(new_users)}{c.E}{c.b} new users...')

    #Set up asynchrounous logic (this works usually, but doesn't for our requests for some reason, I dont know why yet)
    tasks = [asyncio.create_task(scrape(user)) for user in new_users]

    await asyncio.gather(*tasks)


while True:
    start_time = time.time()

    #We can only run asynchrounous functions using this, again I dont know why
    asyncio.run(main())

    #check if we've scraped any ratings
    if len(batch):
        #insert into database
        database['Users'].insert_many(batch)
        print(f'\U00002705 {c.b}Finished scraping. {c.g}{c.B}+{len(batch)}{c.E}{c.b} users\' ratings. Average Speed: {c.v}{c.B}{round(len(batch)/(time.time()-start_time),1)}{c.b} per second.{c.E}')
    else:
        #if none were scraped, continue on
        print(f'{c.y}No ratings were scraped for this batch.{c.E}')

    batch.clear()

    #increase wait time if we get Anti-DoS
    if wait:
        wait_time += 1
        wait = False

    print(f'{c.b}Waiting {c.p}{c.B}{wait_time}{c.E}{c.b} seconds...{c.E}')
    time.sleep(wait_time)




