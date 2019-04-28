import sys
import re
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(engine.table_names()[0],con=engine)
    X = df.message
    y = df.iloc[:,5:]
    category_names = df.columns[5:]
    for col in y.columns:
        y.loc[y[col]>1,col] = 1
        
    return X, y, category_names


def tokenize(text):
    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize sentence
    token = word_tokenize(text)
    # remove stopwords and lemmatize
    words = [WordNetLemmatizer().lemmatize(w) for w in token if w not in stopwords.words('english')]

    return words


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'vect__ngram_range':[(1,2),(2,2)],
                  'clf__estimator__n_estimators':[3, 5]}

    cv = GridSearchCV(estimator = pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
        
    y_preds = model.predict(X_test)
    # print classification report
    print(classification_report(y_preds, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_preds)))


def save_model(model, model_filepath):
    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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