from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>','',str(text)) # HTML 태그들을 삭제
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|p)', text) #이모티콘들은 감정분석에 도움이 되기 때문에 삭제하지 않음
    text = (re.sub('[W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')) #문자나 숫자가 아닌 것을 모두 삭제 후, 이모티콘을 붙임
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)