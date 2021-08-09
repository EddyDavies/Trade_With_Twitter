import os

from pymongo import MongoClient


def extract_env_vars():
    # Get list of dates specified in env vars
    m = os.environ.get("TWITTER_DATE", "Apr 17 Aug 17").split(" ")
    months_list = list(map(' '.join, zip(m[::2], m[1::2])))

    # Get the mongo client link with user and password
    mongo_url = os.environ.get('MONGO_CLIENT_DOWNLOAD', "mongodb://127.0.0.1:2700")
    mongo_user = os.environ.get('MONGO_USER')
    mongo_pwd = os.environ.get('MONGO_PWD')
    # Get mongo client item using username and password if present
    if mongo_user is None:
        mongo = MongoClient(mongo_url)
    else:
        mongo_url = mongo_url.split("://")[0] + "://%s:%s@" + mongo_url.split("://")[1]
        mongo = MongoClient(mongo_url % (mongo_user, mongo_pwd))
    print(mongo_url)

    dbnames = os.environ.get('DBNAMES', "bitcoin ethereum").split(" ")
    dbs = [mongo["shared"]]
    for name in dbnames:
        dbs.append(mongo[name])

    print(f"Date: {months_list[0]} to {months_list[-1]}  DBNames: {dbnames}")

    return months_list, dbs, mongo["bitcoin"], mongo


months, dbs, db, mongo = extract_env_vars()
