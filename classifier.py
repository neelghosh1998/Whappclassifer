
import argparse
import pandas as pd
from textlib.whatsapp import helper
from textlib.whatsapp import general
import emoji as em
from textlib.whatsapp import testhelper
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split
import numpy as np


def parse_arguments() ->  argparse.Namespace:
    """ Parse command line inputs """
    parser = argparse.ArgumentParser(description='Character')
    parser.add_argument('--trainfile', help='The name of the whatsapp export file', required=True)
    parser.add_argument('--testfile', help='The name of the test whatsapp text file with only texts', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Load data
    df = helper.import_data(f'{args.trainfile}')
    df = helper.preprocess_data(df)
    temp=df['Message_Clean']
    l=[]
    for i in temp:
        x=''
        for c in i:
            if c in em.UNICODE_EMOJI:
                x=x+c
        l.append(x)
    df['emojis']=l
    df['Message_Only_Text']=df['Message_Only_Text']+df['emojis']
    dftest= testhelper.import_data(f'{args.testfile}')
    cc=[]
    for i in df['Message_Only_Text']:
        cc.append(i)
    corpus=cc
    y = df['User']
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    tes=dftest['Message_Only_Text']+dftest['emojis']
    test=[tes]
    corpus.append(test[0])
    TV=TfidfVectorizer(max_features = 50)
    XT = TV.fit_transform(corpus).toarray()
    
    
    X=XT[:-1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.05)
    xx=X.tolist()[-1]
    xtest=np.array([xx])
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(xtest)
    name=labelencoder.inverse_transform(y_pred)
    print (name.tolist()[0])
    return 
    
    

    
if __name__ == "__main__":
    main()
