import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import re

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "twitchyrecs"
MYSQL_PORT = 3306
MYSQL_DATABASE = "video_games"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this

# Load games JSON file
f = open('games.json')
games = json.load(f)
games_genre_dict = dict()

for game in games:
    games_genre_dict[game["Title"]] = re.findall(r"'([\w\s]+)'", game["Genres"])

def jaccard_similarity(s1, s2):
    if len(s1) + len(s2) == 0:
        return 0
    numerator = s1.intersection(s2)
    denominator = s1.union(s2)
    return len(numerator) / len(denominator)

def json_search(game_title):
    print(game_title)
    title_results = games_genre_dict.get(game_title, None)
    results_unranked = list()
    if title_results == None:
        return "Game not found"
    for k, v in games_genre_dict.items():
        if k != game_title:
            score = jaccard_similarity(set(title_results), set(v))
            results_unranked.append((score, k))
    results = sorted(results_unranked, key=lambda x: x[0], reverse=True)
    print(results)
    return results


# def sql_search(episode):
#     query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
#     keys = ["id", "title", "descr"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys, i)) for i in data])


@ app.route("/")
def home():
    return render_template('base.html', title="sample html")


@ app.route("/games")
def games_search():
    game_name = request.args.get("game_title")
    return json_search(game_name)


# @ app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return sql_search(text)


app.run(debug=True)
