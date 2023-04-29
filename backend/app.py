import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

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
f = open('games.json', encoding="utf8")
games = json.load(f)

unique_games = {}

games_genre_dict = {}
games_summary_dict = {}
games_reviews_dict = {}
games_rating_dict = {}
games_players_dict = {}
game_title_to_index = {}
game_index_to_title = {}

genres = set()
def tokenize(text):
    return [x for x in re.findall(r"[a-z]+", text.lower())]

unique_game_counter = -1
for game in games:
    title = game["Title"]
    if title not in unique_games: 
        unique_game_counter += 1
        unique_games[title] = game
        games_genre_dict[title] = re.findall(r"'([\w\s]+)'", game["Genres"] if type(game["Genres"]) == str else '')
        genres.update(games_genre_dict[title])

        game_title_to_index[title] = unique_game_counter
        game_index_to_title[unique_game_counter] = title

        games_summary_dict[title] = game["Summary"]
        # games_summary_tokens_dict[game["Title"]] = tokenize(game["Summary"])

        if type(game['Reviews']) == str:
            reviews = re.sub(r'\[|\]', '', game['Reviews'])
            games_reviews_dict[title] = str(reviews)
        else:
            games_reviews_dict[title] = str(game['Reviews'])

        if type(game["Plays"]) != str:
            games_players_dict[title] = game["Plays"]/1000
        else:
            games_players_dict[title] = float(re.sub('K?', "", str(game["Plays"])))
 
        if type(game["Rating"]) == str:
            games_rating_dict[title] = 0.0
        else:
            games_rating_dict[title] = float(game["Rating"])


n_feats = 500
doc_by_vocab = np.empty([len(unique_games), n_feats])

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    # Code from Assignment 5
    return TfidfVectorizer(max_features=max_features, stop_words=stop_words, max_df = max_df, min_df = min_df, norm = norm)


def create_summary_mat():
    tfidf_vec = build_vectorizer(n_feats, "english")
    summary_mat = tfidf_vec.fit_transform([summary for summary in games_summary_dict.values()]).toarray()
    filename = 'sum_mat.pickle'
    pickle.dump(summary_mat, open(filename, "wb"))


def create_reviews_mat():
    tfidf_vec = build_vectorizer(n_feats, "english")
    reviews_mat = tfidf_vec.fit_transform([reviews for reviews in games_reviews_dict.values()]).toarray()
    filename = 'review_mat.pickle'
    pickle.dump(reviews_mat, open(filename, "wb"))


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

    filename = 'game_sims.pickle'
    pickle.dump(game_sims, open(filename, "wb"))   


def get_ranked_games(game_title, req_genre, min_rating, min_players, sim_matrix):
    # Code from Assignment 5
    game_idx = game_title_to_index[game_title]
    
    score_lst = sim_matrix[game_idx]
    game_score_lst = []
    if (req_genre == "Any"):
        req_genre = ""
    for i,s in enumerate(score_lst):
        query_title = game_index_to_title[i]
        query_genre = games_genre_dict[query_title]
        query_rating = games_rating_dict[query_title]
        query_players = games_players_dict[query_title]
        if (req_genre == "" or req_genre in query_genre) and (type(min_rating) == str or query_rating >= min_rating) and (type(min_players) == str or query_players >= min_players):
            game_score_lst.append((query_title, s))

    game_score_lst = game_score_lst[:game_idx] + game_score_lst[game_idx+1:]

    game_score_lst = sorted(game_score_lst, key=lambda x: (-x[1]))

    return game_score_lst


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


def jaccard_similarity(s1, s2):
    if len(s1) + len(s2) == 0:
        return 0
    numerator = s1.intersection(s2)
    denominator = s1.union(s2)
    return len(numerator) / len(denominator)


def cosine_jac_similarity(game_title, req_genre, min_rating, min_players):
    if game_title not in unique_games:
        return "Game not found"

    # create_summary_mat()
    # create_reviews_mat()
    # games_sim_cos_jac = build_game_sims_cos_jac(len(unique_games), game_index_to_title, sum_mat, game_title_to_index, get_cosine_sim, jaccard_similarity)

    sum_mat = pickle.load(open("sum_mat.pickle", "rb"))
    rev_mat = pickle.load(open("review_mat.pickle", "rb"))
    games_sim_cos_jac = pickle.load(open("game_sims.pickle", "rb"))

    ranked_games = get_ranked_games(game_title, req_genre, min_rating, min_players, games_sim_cos_jac)

    if (len(ranked_games) > 10):
        top_games = [game[0] for game in ranked_games[:10]]
    else:
        top_games = [game[0] for game in ranked_games]
    return top_games

@ app.route("/")
def home():
    return render_template('base.html')


@app.route("/games/")
def games_search():
    body = request.args
    game_name = body.get("game_title")
    req_genre = body.get("game_genre")
    min_rating = body.get("game_rating")
    min_players = body.get("game_players")
    try:
        min_rating = float(min_rating)
    except:
        pass 
    try:
        min_players = float(min_players)
    except:
        pass
    return json.dumps(cosine_jac_similarity(game_name, req_genre, min_rating, min_players))

# app.run(debug=True)
