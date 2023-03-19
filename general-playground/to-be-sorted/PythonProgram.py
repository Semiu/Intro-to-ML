# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:30:52 2020

@author: Semiu
"""
#import urllib.parse 
import urllib
from pymongo import MongoClient

# pprint library is used to make the output look more pretty
from pprint import pprint

#username
#username = urllib.parse.quote_plus('taskia')

#password
#password = urllib.parse.quote_plus('AKANMUsemiu'+'%64'+'1984')

URI = "mongodb+srv://taskia:" + 'AKANMUsemiu'+urllib.parse.quote('@') +"--MongodB URI---"

                                                        # connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient(URI)

db=client.admin

# Issue the serverStatus command and print the results
serverStatusResult=db.command("serverStatus")

print("----------print here--------")
pprint(serverStatusResult)


# mongodb+srv://taskia:AKANMUsemiu@1984@cluster0.z1egk.mongodb.net/vuldB?retryWrites=true&w=majority