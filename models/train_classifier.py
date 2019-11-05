# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql('twt_disaster_clean',con=engine)
    X = df[['message']]
    Y = df.drop(['id','message','original','genre','categories'],axis = 1)
    print(Y.head())
    Y.fillna(0,inplace=True)
    for col in Y:
        Y[col] = Y[col].astype(int)
    return X,Y,Y.columns
    


def tokenize(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    clean_tokens = []
    for words in tokens:
        clean_tok = stemmer.stem(words).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('clf',MultiOutputClassifier(RandomForestClassifier()))
                    ])
    
    parameters = {
    'clf__estimator__n_estimators': [10]
#     'clf__estimator__criterion': ['gini', 'entropy'],
#     'clf__estimator__max_features': [5, 10, 15]
    }
    cv = GridSearchCV(pipeline,param_grid=parameters)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    output = model.predict(X_test['message'])
    output = pd.DataFrame(output,columns=category_names)
    for col in output:
        output[col] = output[col].astype(int)


    for col in category_names:
        y_true = Y_test[col]
        y_pred = output[col]
        print(col,classification_report(y_true,y_pred))  


def save_model(model, model_filepath):
    # save the model to disk
    filename = str(model_filepath)
    pickle.dump(model,open(filename,'wb'))
        


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    