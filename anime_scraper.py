#This file was used to scrape animes from MyAnimeList from their webpages

#CONFIG
size = 290 #number of websites to scrape at a time before uploading it to mongodb database
wait_time = 65 #number of seconds to wait between scraping (to avoid Anti-DoS)
##END CONFIG

from pymongo import MongoClient
import pymongo
from requests_html import HTMLSession
from requests_html import AsyncHTMLSession
import json
import pandas as pd
import time
import sys

#connect to database in mongodb
client = MongoClient("INSERT MONGDB ADDRESS HERE",tlsAllowInvalidCertificates=True)
database = client['MyAnimeList']

#colors r,y,g,b,p,v,B,U,E
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

#get the anime_id we've last scraped
last_entry = database['Misc'].find({},{'_id':0})[0]['last_entry']
print(f'{c.v}Last Anime ID scraped: {c.B}{c.p}{last_entry}{c.E}')

#this allows us to scrape multiple websites at a time
asession = AsyncHTMLSession()

#scraped data
batch = []

#this will be True when Anti-DoS is triggered
wait = False

async def scrape(id):

    #visit anime website
    site = await asession.get(f'https://myanimelist.net/anime/{id}')

    #Get the title, and check if page is 404
    try:
        title = site.html.find('#contentWrapper')[0].find('h1')[0].text
        if title == '404 Not Found':
            return
    except Exception as e:
        #Anti-DoS is triggered
        global wait
        if not wait:
            print(f'{c.r}Anti-DoS triggered! \U0001f612 Anime #{id} Error: {e}{c.E}')
        wait = True
        return

    data = {}
    data['_id'] = id
    data['title'] = title
    data['rating'] = site.html.find('.score-label')[1].text

    #get anime tags
    tags = site.html.find('#content .leftside .spaceit_pad')
    for div in tags:

        #div is <div class='spaceit_pad'> element

        tag = div.find('span')[0].text[:-1]
        if tag in ['Type','Status','Rating']:
            data[tag] = div.text[len(tag)+2:]

        #Some tags are plural
        elif tag in ['Genres','Themes','Producers','Studios','Demographics']:
            data[tag] = [x.text for x in div.find('a')]
        elif tag in ['Genre','Theme','Producer','Studio','Demographic']:
            data[tag+'s'] = div.find('a')[0].text
    
    #Get the plot summary
    data['plot'] = site.html.find('p[itemprop="description"]')[0].text.replace('\n','').replace('[Written by MAL Rewrite]','')

    #add the data to batch
    batch.append(data)

#time how long we scraped
start_time = time.time()

while last_entry < 52000:
    print(f'{c.b}Scraping from: {c.B}{c.p}{last_entry}{c.E} \U0001f916')

    #time how long we scraped a batch
    batch_time = time.time()

    #scrape the websites
    asession.run(*[lambda x=y: scrape(x) for y in range(last_entry,last_entry+size)])

    #Anti-DoS is triggered, wait 5 mins
    if wait:
        batch.clear()
        wait = False
        print(f'{c.y}Anti-DoS triggered, restarting in 5 mins...{c.E}')
        time.sleep(60)
        print(f'{c.y}Anti-DoS triggered, restarting in 4 mins...{c.E}')
        time.sleep(60)
        print(f'{c.y}Anti-DoS triggered, restarting in 3 mins...{c.E}')
        time.sleep(60)
        print(f'{c.y}Anti-DoS triggered, restarting in 2 mins...{c.E}')
        time.sleep(60)
        print(f'{c.y}Anti-DoS triggered, restarting in 1 mins...{c.E}')
        time.sleep(60)
        continue

    #check if we've scraped any animes
    if len(batch):
        print(f'{c.g}Batch finished \U00002705 {c.b}Average Speed: {c.p}{c.B}{round(size/(time.time()-batch_time))}{c.E}{c.b} per second. {c.g}{c.B}+{len(batch)}{c.E}{c.b} animes{c.E}')

        #insert into database
        database['Animes'].insert_many(batch)
        batch.clear()

    else:
        #if no animes were scraped, continue on
        batch.clear()
        print(f'{c.y}No Animes were scraped for this batch.{c.E}')


    #update last_entry to keep track of the last anime we've scraped
    last_entry += size
    database['Misc'].update_one({},{ "$set": { "last_entry": last_entry} })

    #waiting wait_time to not trigger anti-DoS.
    print(f'{c.b}Last Entry is now: {c.B}{c.p}{last_entry}{c.E}{c.b} Waiting {c.B}{c.p}{wait_time}{c.E}{c.b} seconds...{c.E}')
    time.sleep(wait_time)



