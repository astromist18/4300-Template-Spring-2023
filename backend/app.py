import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "twitchrecs"
MYSQL_PORT = 3306
MYSQL_DATABASE = "video_games"

# mysql_engine = MySQLDatabaseHandler(
#     MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Load games JSON file
f = open(os.path.join(os.environ['ROOT_PATH'], 'games.json'), encoding="utf8")
games = json.load(f)

unique_games = {}

games_genre_dict = {}
games_summary_reviews_dict = {}
games_summary_reviews_tokens_dict = {}
game_title_to_index = {}
game_index_to_title = {}

def tokenize(text):
    return [x for x in re.findall(r"[a-z]+", text.lower())]

unique_game_counter = -1
for game in games:
    if game["Title"] not in unique_games: 
        unique_game_counter += 1
        unique_games[game["Title"]] = game
        games_genre_dict[game["Title"]] = re.findall(r"'([\w\s]+)'", game["Genres"] if type(game["Genres"]) == str else '')

        game_title_to_index[game["Title"]] = unique_game_counter
        game_index_to_title[unique_game_counter] = game["Title"]

        if type(game['Reviews']) == str:
            reviews = re.split('\', \'|\", \"|\", \'|\', \"', game['Reviews'].strip(']['))
            games_summary_reviews_dict[game["Title"]] = reviews
        else:
            games_summary_reviews_dict[game["Title"]] = game['Reviews']
        games_summary_reviews_dict[game["Title"]].insert(0, game["Summary"])

        tokens = []
        for text in games_summary_reviews_dict[game["Title"]]:
            tokens.append(tokenize(text))

        games_summary_reviews_tokens_dict[game["Title"]] = tokens

n_feats = 5000
doc_by_vocab = np.empty([len(unique_games), n_feats])

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    # Code from Assignment 5
    return TfidfVectorizer(max_features=max_features, stop_words=stop_words, max_df = max_df, min_df = min_df, norm = norm)

def create_mat():
    tfidf_vec = build_vectorizer(n_feats, "english")
    return tfidf_vec.fit_transform([' '.join(summary_reviews) for summary_reviews in games_summary_reviews_dict.values()]).toarray()

def get_cosine_sim(game1, game2, input_doc_mat, input_game_title_to_index):
    # Code from Assignment 5
    game1_idx = input_game_title_to_index[game1]
    game2_idx = input_game_title_to_index[game2]
    
    q = input_doc_mat[game1_idx]
    d = input_doc_mat[game2_idx]
   
    sim = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
    
    if np.isnan(sim):
        sim = 0

    return sim

def build_game_sims_cos_jac(n_games, game_index_to_title, input_doc_mat, game_title_to_index, input_get_cos_sim_method, input_get_jac_sim_method):
    # Code from Assignment 5
    game_sims = np.zeros((n_games, n_games))
    for i in range(0, n_games):
        for j in range(i, n_games):
            game1 = game_index_to_title[i]
            game2 = game_index_to_title[j]
            cos_sim = input_get_cos_sim_method(game1, game2, input_doc_mat, game_title_to_index)
            jac_sim = input_get_jac_sim_method(set(games_genre_dict.get(game1, [])), set(games_genre_dict.get(game2, [])))
            sim = jac_sim**0.4 + cos_sim*2
            game_sims[i, j] = game_sims[j, i] = sim
                
    return game_sims


def get_ranked_games(game, matrix):
    # Code from Assignment 5
    game_idx = game_title_to_index[game]
    
    score_lst = matrix[game_idx]
    game_score_lst = [(game_index_to_title[i], s) for i,s in enumerate(score_lst)]
    
    game_score_lst = game_score_lst[:game_idx] + game_score_lst[game_idx+1:]
    
    game_score_lst = sorted(game_score_lst, key=lambda x: (-x[1]))

    return game_score_lst


def jaccard_similarity(s1, s2):
    if len(s1) + len(s2) == 0:
        return 0
    numerator = s1.intersection(s2)
    denominator = s1.union(s2)
    return len(numerator) / len(denominator)


def json_search(game_title):
    title_results = games_genre_dict.get(game_title, None)
    results_unranked = list()
    if title_results == None:
        return "Game not found"
    for k, v in games_genre_dict.items():
        if k != game_title:
            score = jaccard_similarity(set(title_results), set(v))
            results_unranked.append((score, k))
    results = sorted(results_unranked, key=lambda x: x[0], reverse=True)
    games = [x[1] for x in results]
    return games

def cosine_jac_similarity(game_title):
    if game_title not in unique_games:
        return "Input a valid game name"

    doc_by_vocab = create_mat()

    games_sim_cos_jac = build_game_sims_cos_jac(len(unique_games), game_index_to_title, doc_by_vocab, game_title_to_index, get_cosine_sim, jaccard_similarity)

    ranked_games = get_ranked_games(game_title, games_sim_cos_jac)

    top_games = [game[0] for game in ranked_games[:10]]
    return top_games

@ app.route("/")
def home():
    body = request.args.get("game_name")
    sim = cosine_jac_similarity(body)
    most_sim = []
    for i in range(0, 5):
        most_sim.append(sim[i])
    if (sim == "Input a valid game name"):
        most_sim = "Input a valid game name"
        game = ""
    else:
        game = "Your game: " + body
    return render_template('base.html', title="sample html", game=game, similarity=most_sim)


@app.route("/games/")
def games_search():
    body = request.args
    game_name = body["game_title"].capitalize()
    game_genre = body["game_genre"].capitalize()
    game_rating = body["game_rating"]
    game_players = body["game_players"]
    return json.dumps(cosine_jac_similarity(game_name))

app.run(debug=True)
