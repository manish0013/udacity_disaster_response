import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)

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
df = pd.read_sql('twt_disaster_clean', con = engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    def get_top_n_words(corpus, n=None):
        """
        To get top n words from a corpus of text
        Args 
        corpus = raw text 
        n = top n words
        
        returns a list of key value pairs consisting of top words and frequency
        """
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    stop = text.ENGLISH_STOP_WORDS
    # corpus to be used to identify top n words
    corpus_news=   df[df['genre'] == 'news']['message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    corpus_direct= df[df['genre'] == 'direct']['message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    corpus_social= df[df['genre'] == 'social']['message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    top_news =   get_top_n_words(corpus_news,n = 10)
    top_direct = get_top_n_words(corpus_direct,n = 10)
    top_social = get_top_n_words(corpus_social,n = 10)

    news_index = []
    news_values = []
    for i in range(10):
        news_index.append(top_news[i][0])
        news_values.append(top_news[i][1])

    direct_index = []
    direct_values = []
    for i in range(10):
        direct_index.append(top_direct[i][0])
        direct_values.append(top_direct[i][1])

    social_index = []
    social_values = []
    for i in range(10):
        social_index.append(top_social[i][0])
        social_values.append(top_social[i][1])
      

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_index,
                    y=news_values
                )
            ],

            'layout': {
                'title': 'Top 10 words in news Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_index,
                    y=direct_values
                )
            ],

            'layout': {
                'title': 'Top 10 words in direct Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_index,
                    y=social_values
                )
            ],

            'layout': {
                'title': 'Top 10 Words in Social Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }



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
