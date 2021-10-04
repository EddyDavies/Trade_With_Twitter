import os
from typing import List, Optional

from pymongo import MongoClient
from pydantic import BaseSettings, Field


class MongoSettings(BaseSettings):
    # Get the mongo client link and database name
    db_name = "bitcoin"
    url: str
    user = Optional[str]
    pwd = Optional[str]

    # Get mongo client item using username and password if present
    if user:
        url = url.split("://")[0] + "://%s:%s@" + url.split("://")[1]
        url = url % (user, pwd)
    client = MongoClient(url % (user, pwd))
    db = client[db_name]

    class Config:
        env_prefix = 'mongo_'
        fields = {
            'url': {
                'env': ['MONGO_CLIENT', 'MONGO_CLIENT_DOWNLOAD'],
            }
        }


class RunSettings(BaseSettings):
    twitter_date: List[str]

    # Get list of dates specified in env vars
    m = os.environ.get("TWITTER_DATE", "Apr 17 Aug 17").split(" ")
    months = list(map(' '.join, zip(m[::2], m[1::2])))


months, db, mongo = RunSettings.months, MongoSettings.db, MongoSettings.client

print(MongoSettings.url)
print(f"Date: {RunSettings.months[0]} to {RunSettings.months[-1]}  DBNames: {MongoSettings.db_name}")
