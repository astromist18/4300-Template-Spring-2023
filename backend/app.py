import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from textblob import TextBlob

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

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
games_sentiment_dict = {}
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
        
        blob = TextBlob(games_reviews_dict[title])
        games_sentiment_dict[title] = blob.sentiment.polarity

        if type(game["Plays"]) != str:
            games_players_dict[title] = game["Plays"]/1000
        else:
            games_players_dict[title] = float(re.sub('K?', "", str(game["Plays"])))
 
        if type(game["Rating"]) == str:
            games_rating_dict[title] = 0.0
        else:
            games_rating_dict[title] = float(game["Rating"])


n_feats = 5000
doc_by_vocab = np.empty([len(unique_games), n_feats])

def build_vectorizer(max_features, stop_words, norm='l2'):
    # Code from Assignment 5
    return TfidfVectorizer(max_features=max_features, stop_words=stop_words, norm = norm)

tfidf_vec = build_vectorizer(n_feats, "english")

def create_summary_mat():
    summary_mat = tfidf_vec.fit_transform([summary for summary in games_summary_dict.values()]).toarray()
    filename = 'sum_mat.pickle'
    pickle.dump(summary_mat, open(filename, "wb"))


def create_reviews_mat():
    reviews_mat = tfidf_vec.fit_transform([reviews for reviews in games_reviews_dict.values()]).toarray()
    filename = 'review_mat.pickle'
    pickle.dump(reviews_mat, open(filename, "wb"))


def build_game_sims_cos_jac(n_games, game_index_to_title, input_sum_doc_mat, input_rev_doc_mat, game_title_to_index, input_get_cos_sim_method, input_get_jac_sim_method):
    # Code from Assignment 5
    game_sims = np.zeros((n_games, n_games))
    for i in range(0, n_games):
        for j in range(i, n_games):
            game1 = game_index_to_title[i]
            game2 = game_index_to_title[j]
            sum_cos_sim = input_get_cos_sim_method(game1, game2, input_sum_doc_mat, game_title_to_index)
            rev_cos_sim = input_get_cos_sim_method(game1, game2, input_rev_doc_mat, game_title_to_index)
            cos_sim = sum_cos_sim*.6 + rev_cos_sim*.4
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
    # create_summary_mat()
    # create_reviews_mat()
    # sum_mat = pickle.load(open("sum_mat.pickle", "rb"))
    # rev_mat = pickle.load(open("review_mat.pickle", "rb"))

    # build_game_sims_cos_jac(len(unique_games), game_index_to_title, sum_mat, rev_mat, game_title_to_index, get_cosine_sim, jaccard_similarity)
    games_sim_cos_jac = pickle.load(open("game_sims.pickle", "rb"))

    ranked_games = get_ranked_games(game_title, req_genre, min_rating, min_players, games_sim_cos_jac)
    # print(ranked_games[:10])

    if (len(ranked_games) > 10):
        top_games = [game[0] for game in ranked_games[:10]]
    else:
        top_games = [game[0] for game in ranked_games]
    similarities = [games_sentiment_dict[game] for game in top_games]
    sim_converted = ['🙂🙂' if s >= 0.5 else '🙂' if s > 0.05 else '🙁🙁' if s < -0.5 else '🙁' if s < -0.05 else '😐' for s in similarities]
    return (top_games, sim_converted)

@ app.route("/")
def home():
    game_titles = list(game_title_to_index.keys())
    return render_template('base.html', titles=game_titles)


@app.route("/games/")
def games_search():
    body = request.args
    game_name = body.get("game_title")
    req_genre = body.get("game_genre")
    min_rating = body.get("game_rating")
    min_players = body.get("game_players")

    if game_name not in unique_games:
        return json.dumps("Game not found")

    try:
        min_rating = float(min_rating)
    except:
        pass 
    try:
        min_players = float(min_players)
    except:
        pass
    top_games, similarities = cosine_jac_similarity(game_name, req_genre, min_rating, min_players)
    return json.dumps((top_games, similarities))

@app.route("/features/")
def get_features():
    body = request.args
    k = int(body.get("num_features"))
    input_title = body.get("input_title")
    result_title_str = body.getlist("result_titles")[0]

    sum_mat = tfidf_vec.fit_transform([summary for summary in games_summary_dict.values()]).toarray()
    input_doc_idx = game_title_to_index[input_title]
    feature_array = list(tfidf_vec.get_feature_names())

    # input_title_sum_top_k_tfidf_scores = {}
    input_title_top_k_tfidf_scores = sorted(sum_mat[input_doc_idx], reverse=True)[:k]
    used_idx = set()
    input_title_top_k_features = []
    print('index: ', input_doc_idx)
    for top_score in input_title_top_k_tfidf_scores:
        for idx, score in enumerate(list(sum_mat[input_doc_idx])):
            if score == top_score and idx not in used_idx:
                used_idx.add(idx)
                input_title_top_k_features.append(feature_array[idx])
                break
    # input_title_top_k_features = [feature_array[list(sum_mat[input_doc_idx]).index(score)] 
    #                               for score in input_title_top_k_tfidf_scores]
    print(input_title_top_k_tfidf_scores)
    print(input_title_top_k_features)
    print(list(sum_mat[input_doc_idx]).index(sorted(sum_mat[input_doc_idx], reverse=True)[0]))

    result_titles_top_k_features = []
    result_titles = result_title_str.split(",")
    for result_title in result_titles:
        sum_mat = tfidf_vec.fit_transform([summary for summary in games_summary_dict.values()]).toarray()
        input_doc_idx = game_title_to_index[result_title]
        feature_array = list(tfidf_vec.get_feature_names())

        result_title_top_k_tfidf_scores = sorted(sum_mat[input_doc_idx], reverse=True)[:k]
        used_idx = set()
        result_title_top_k_features = []
        print('index: ', input_doc_idx)
        for top_score in result_title_top_k_tfidf_scores:
            for idx, score in enumerate(list(sum_mat[input_doc_idx])):
                if score == top_score and idx not in used_idx:
                    used_idx.add(idx)
                    result_title_top_k_features.append(feature_array[idx])
                    break

        result_titles_top_k_features.append(result_title_top_k_features)

    return [input_title_top_k_features, result_titles_top_k_features]

# app.run(debug=True)
