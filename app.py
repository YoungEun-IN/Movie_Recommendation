#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser

import flask
import pandas as pd
import pymysql
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')

# df = pd.read_csv('./model/tmdb.csv')

# 설정 파일 경로
config_path = "./setting.ini"

# 설정 파일 읽기
config = configparser.ConfigParser(interpolation=None)
config.read(config_path)

# params
user = config['setting']['user']
password = config['setting']['password']
host = config['setting']['host']
port = int(config['setting']['port'])
database = config['setting']['database']

conn = pymysql.connect(host=host, user=user, db=database, port=port, password=password, charset='utf8')

sql = "select * from tmdb"

df = pd.read_sql_query(sql, conn)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])
all_titles = [df['title'][i] for i in range(len(df['title']))]


def get_recommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = df['title'].iloc[movie_indices]
    dat = df['release_date'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title', 'Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return return_df


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
        #        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if m_name not in all_titles:
            return flask.render_template('negative.html', name=m_name)
        else:
            result_final = get_recommendations(m_name)
            names = []
            dates = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])

            return flask.render_template('positive.html', movie_names=names, movie_date=dates, search_name=m_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
