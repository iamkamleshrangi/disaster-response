import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from plotly.graph_objs import Bar, Pie


app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ["VB", "VBP"] or first_word == "RT":
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = load("../models/classifier.joblib")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    category_counter = df.drop(["id", "message", "original", "genre"], axis=1).sum()
    category_counter.sort_values(ascending=False, inplace=True)
    category_names = list(category_counter.index)

    # create visuals
    graphs = [
        {
            "data": [
                Pie(
                    hole=0.4,
                    name="Genre",
                    pull=0,
                    marker=dict(colors=["#9AD59A", "#B4B8D0", "#F5BE51"]),
                    textinfo="label+percent+value",
                    hoverinfo="all",
                    labels=genre_names,
                    values=genre_counts,
                    uid="95cd0b0c",
                )
            ],
            "layout": {
                "title": "Distribution of Message Genres",
                "titlefont": {"size": 18},
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [
                Bar(x=category_names, y=category_counter, marker=dict(color="#463659"))
            ],
            "layout": {
                "title": "Distribution of Message Categories",
                "titlefont": {"size": 18},
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category", "tickangle": 35, "automargin": True},
            },
        },
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
