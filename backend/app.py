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
f = open(os.path.join(os.environ['ROOT_PATH'], 'games.json'), encoding="utf8")
games = json.load(f)

games_genre_dict = {}
games_summary_reviews_dict = {}
games_summary_reviews_tokens_dict = {}

game_title_to_index = {game_title:index for index, game_title in enumerate([game["Title"] for game in games])}
game_index_to_title = {v:k for k,v in game_title_to_index.items()}

def tokenize(text):
    return [x for x in re.findall(r"[a-z]+", text.lower())]

for game in games:
    games_genre_dict[game["Title"].lower()] = re.findall(r"'([\w\s]+)'", game["Genres"] if type(game["Genres"]) == str else '')

    if type(game['Reviews']) == str:
        reviews = re.split('\', \'|\", \"|\", \'|\', \"', game['Reviews'].strip(']['))
        games_summary_reviews_dict[game["Title"].lower()] = reviews
    else:
        games_summary_reviews_dict[game["Title"].lower()] = game['Reviews']
    games_summary_reviews_dict[game["Title"].lower()].insert(0, game["Summary"])

    tokens = []
    for text in games_summary_reviews_dict[game["Title"].lower()]:
        tokens.append(tokenize(text))

    games_summary_reviews_tokens_dict[game["Title"].lower()] = tokens

# def build_inverted_idx(text_tokens):
#     inverted_index = {}
#     text_counter = 0
#     for text in text_tokens:
#         token_counter = {}
#         for t in text:
#             if t in token_counter:
#                 token_counter[t] += 1
#             else:
#                 token_counter[t] = 1
                
#         for k, v in token_counter.items():
#             if k in inverted_index:
#                 inverted_index[k].append((text_counter, v))
#             else:
#                 inverted_index[k] = [(text_counter, v)]
                
#         text_counter += 1
        
#     return inverted_index

# def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
#     idf = {}
#     for k, v in inv_idx.items():
#         df = len(v)
#         df_ratio = df/n_docs
#         if df > min_df and df_ratio < max_df_ratio:
#             idf[k] = math.log2(n_docs/(1+df))
        
#     return idf

n_feats = 5000
doc_by_vocab = np.empty([len(games), n_feats])

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    # Code from Assignment 5
    return TfidfVectorizer(max_features=max_features, stop_words=stop_words, max_df = max_df, min_df = min_df, norm = norm)

tfidf_vec = build_vectorizer(n_feats, "english")
doc_by_vocab = tfidf_vec.fit_transform([' '.join(summary_reviews) for summary_reviews in games_summary_reviews_dict.values()]).toarray()
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}

def cosine_sim(game1, game2, input_doc_mat, input_game_title_to_index):
    # Code from Assignment 5
    game1_idx = input_game_title_to_index
    game2_idx = input_game_title_to_index
    
    q = input_doc_mat[game1_idx]
    d = input_doc_mat[game2_idx]
    
    sim = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
    
    return sim

def build_game_sims_cos(n_games, game_index_to_title, input_doc_mat, game_title_to_index, input_get_sim_method):
    # Code from Assignment 5
    game_sims = np.zeros((n_games, n_games))
    for i in range(0, n_games):
        for j in range(i, n_games):
            game1 = game_index_to_title[i]
            game2 = game_index_to_title[j]
            sim = input_get_sim_method(game1, game2, input_doc_mat, game_title_to_index)
            game_sims[i, j] = game_sims[j, i] = sim
            
    return game_sims


def get_ranked_movies(game, matrix):
    # Code from Assignment 5
    game_idx = game_title_to_index[game]
    
    score_lst = matrix[game_idx]
    game_score_lst = [(game_index_to_title[i], s) for i,s in enumerate(score_lst)]
    
    game_score_lst = game_score_lst[:game_idx] + game_score_lst[game_idx+1:]
    
    game_score_lst = sorted(game_score_lst, key=lambda x: -x[1])
    
    return game_score_lst

def jaccard_similarity(s1, s2):
    if len(s1) + len(s2) == 0:
        return 0
    numerator = s1.intersection(s2)
    denominator = s1.union(s2)
    return len(numerator) / len(denominator)

def json_search(game_title):
    print(game_title)
    title_results = games_genre_dict.get(game_title.lower(), None)
    results_unranked = list()
    if title_results == None:
        print("Game not found")
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
    game_name = request.args.get("game_title").capitalize()
    return json_search(game_name)


# @ app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return sql_search(text)


app.run(debug=True)
