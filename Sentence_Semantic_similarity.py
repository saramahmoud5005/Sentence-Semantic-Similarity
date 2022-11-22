#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install gensim')
import gensim
import gensim.downloader as api
from scipy import spatial
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


# Load data set and print head

# In[59]:


data = pd.read_csv("E:/6th Term/Natural Language Processing/project/train.csv")

print("Number of data points:",data.shape[0])
data.head()


# display info of data

# In[60]:


data.info()


# **Visualization of data set**

# In[61]:


data.groupby("is_duplicate")['id'].count().plot.bar()


# In[62]:


qs_series = pd.Series(data['qid1'].tolist() + data['qid2'].tolist())
num_Unique_qs = len(np.unique(qs_series))
qs_appear_morethan_onetime = np.sum(qs_series.value_counts() > 1)
print ('Total number of  Unique Questions are: {}\n'.format(num_Unique_qs))

print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_appear_morethan_onetime,qs_appear_morethan_onetime/num_Unique_qs*100))

print ('Max number of times a single question is repeated: {}\n'.format(max(qs_series.value_counts()))) 

q_vals=qs_series.value_counts()

q_vals=q_vals.values


# In[63]:


a = ["unique_questions" , "Repeated Questions"]
b =  [num_Unique_qs , qs_appear_morethan_onetime]

plt.figure(figsize=(10, 6))
plt.title ("Plot representing unique and repeated questions  ")
sns.barplot(a,b)
plt.show()


# **Preprocessing**

# In[64]:


#check nulls
data['question1'].fillna("", inplace = True)
data['question2'].fillna("", inplace = True)
print(data.isnull().sum())
print(data.isnull().values.sum())
f, ax = plt.subplots(figsize=(50, 6))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap="viridis",ax=ax)


# In[65]:


def replace_abbreviation(sentence):
  sentence = re.sub(r"1st", "first", sentence)
  sentence = re.sub(r"2nd", "second", sentence)
  sentence = re.sub(r"3rd", "third", sentence)
  # sentence = re.sub(r'([0-9]+)k ', r'\1000 ', sentence)
  # sentence = re.sub(r'([0-9]+)m ', r'\1000000 ', sentence)
  # sentence = re.sub(r'([0-9]+)b ', r'\1000000000 ', sentence)
  sentence = re.sub(r'([0-9]+)y ', r'\1 years ', sentence)
  sentence = re.sub(r'([0-9]+)s ', r'\1 seconds ', sentence)
  sentence = re.sub(r"what's", "", sentence)
  sentence = re.sub(r"\'s", " ", sentence)
  sentence = re.sub(r"\'ve", " have ", sentence)
  sentence = re.sub(r"can't", "can not ", sentence)
  sentence = re.sub(r"n't", " not ", sentence)
  sentence = re.sub(r"i'm", "i am", sentence)
  sentence = re.sub(r" m ", " am ", sentence)
  sentence = re.sub(r"\'re", " are ", sentence)
  sentence = re.sub(r"\'d", " would ", sentence)
  sentence = re.sub(r"\'ll", " will ", sentence)
  sentence = re.sub(r" e g ", " eg ", sentence)
  sentence = re.sub(r" b g ", " bg ", sentence)
  sentence = re.sub(r"\0s", "0", sentence)
  sentence = re.sub(r" 9 11 ", "911", sentence)
  sentence = re.sub(r"e-mail", "email", sentence)
  sentence = re.sub(r"\s{2,}", " ", sentence)
  sentence = re.sub(r"quikly", "quickly", sentence)
  sentence = re.sub(r" usa ", " america ", sentence)
  sentence = re.sub(r" u s ", " america ", sentence)
  sentence = re.sub(r" uk ", " england ", sentence)
  sentence = re.sub(r"imrovement", "improvement", sentence)
  sentence = re.sub(r"intially", "initially", sentence)
  sentence = re.sub(r" dms ", "direct messages ", sentence)  
  sentence = re.sub(r"demonitization", "demonetization", sentence) 
  sentence = re.sub(r"actived", "active", sentence)
  sentence = re.sub(r"kms", " kilometers ", sentence)
  sentence = re.sub(r" cs ", " computer science ", sentence) 
  sentence = re.sub(r" upvotes ", " up votes ", sentence)
  sentence = re.sub(r" iphone ", " phone ", sentence)
  sentence = re.sub(r"\0rs ", " rs ", sentence) 
  sentence = re.sub(r"calender", "calendar", sentence)
  sentence = re.sub(r"ios", "operating system", sentence)
  sentence = re.sub(r"programing", "programming", sentence)
  sentence = re.sub(r"bestfriend", "best friend", sentence)
  sentence = re.sub(r"iii", "3", sentence) 
  sentence = re.sub(r"the us", "america", sentence)
  sentence = re.sub(r" j k ", " jk ", sentence)
  sentence = re.sub(r"[^A-Za-z0-9]", " ", sentence)

  return sentence


# In[66]:


def sentence_preprocess(sentence):
  
  # Change sentence to lowercase
  sentence = sentence.lower()
  # Replace abbreviations with its original 
  sentence = replace_abbreviation(sentence)

  # # Remove punctuation
  # for c in sentence:
  #   if c in punctuation:
  #     sentence = sentence.replace(c, " ")
  
  # Tokenization
  tokens = word_tokenize(sentence)

  # Remove stop words
  stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']
        
  for token in list(tokens):
    if token in stop_words:
      tokens.remove(token)
  
  lemmatizer=WordNetLemmatizer()


  
  tokens = [lemmatizer.lemmatize(token) for token in list(tokens)]

  sentence = ""
  for token in list(tokens):
    sentence += token
    sentence += " "
  return sentence


# In[67]:


for i in range(404290):
  data['question1'][i] = sentence_preprocess(data['question1'][i])
  data['question2'][i] = sentence_preprocess(data['question2'][i])


# In[68]:


preprocessed_data = pd.DataFrame(data)
preprocessed_data.to_csv('preprocessed Data.csv', index = False)


# In[69]:


preprocessed_data.head()


# Feature extraction after preprocessing

# In[70]:


new_data=pd.read_csv("preprocessed Data.csv")
new_data.head()


# In[71]:


new_data.fillna(" ", inplace = True)
print(new_data.isnull().values.sum())


# In[72]:


#get unique qids after preprocessing
qids = pd.Series(new_data['qid1'].tolist() + new_data['qid2'].to_list())
len_unique_qs = len(np.unique(qids))
unique_qs=np.unique(qids)
len_unique_qs


# In[73]:


#get unique id with its questions
q1=new_data.drop(["qid2","id","is_duplicate","question2"],axis=1)
q1.drop_duplicates(inplace=True)
q1.rename(columns = {'qid1':'id','question1':'sentence'}, inplace = True)

q2=new_data.drop(["qid1","id","is_duplicate","question1"],axis=1)
q2.drop_duplicates(inplace=True)
q2.rename(columns = {'qid2':'id','question2':'sentence'}, inplace = True)

questions=q1.append(q2)
questions.drop_duplicates(inplace=True)
questions.head()


# In[74]:


#toknize unique questions to use it in model
list_of_sentences=questions['sentence'].tolist()
tags=questions["id"].tolist()
list_of_toknized=[]

for i,item in enumerate(list_of_sentences):   
     list_of_toknized.append(list((word_tokenize(item))))


# **Use Doc2vec model to get similarity vector for each quetions**
# 
# 
# * Note:doc2vec work like word2vec but it take sentences or paragraph instead of words
# 
# 
# 

# In[75]:


def tagged_document(list_of_list_of_words,IDs):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [IDs[i]])
      
data_for_training = list(tagged_document(list_of_toknized,tags))      


# In[76]:


print(data_for_training[:1])


# In[77]:


model = gensim.models.doc2vec.Doc2Vec(vector_size=60, min_count=1,epochs=30)
model.build_vocab(data_for_training)
model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)
model.docvecs[1]


# In[78]:


cosine_similarity=[]
def get_cosine_similarity(row):
    #print(row["qid1"])
    sentence1_vec=model.docvecs[row["qid1"]]
    sentence2_vec=model.docvecs[row["qid2"]]
    return 1 - spatial.distance.cosine(sentence1_vec, sentence2_vec)


# In[79]:


new_data["cosine_similarity"]=new_data.apply(get_cosine_similarity,axis=1)


# In[80]:


def common_words_ratio(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    m=len(w1)+len(w2)    
    return (len(w1 & w2))/m


# In[81]:


new_data["word_share"]=new_data.apply(common_words_ratio,axis=1)


# In[82]:



def fetch_token_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    SAFE_DIV = 0.0001 
 
    token_features = [0.0]*8
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[1] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
   
    # Last word of both question is same or not
    token_features[2] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[3] = int(q1_tokens[0] == q2_tokens[0])

    
    

    return token_features


# In[83]:


token_features = new_data.apply(fetch_token_features, axis=1)

new_data["cwc_min"]       = list(map(lambda x: x[0], token_features))
new_data["cwc_max"]       = list(map(lambda x: x[1], token_features))
new_data["last_word_eq"]  = list(map(lambda x: x[2], token_features))
new_data["first_word_eq"] = list(map(lambda x: x[3], token_features))


# In[84]:


new_data['q1_freq']=new_data.groupby('qid1')['qid1'].transform('count')
new_data['q2_freq']=new_data.groupby('qid2')['qid2'].transform('count')


# In[85]:


new_data.head(10)


# split data into 80% train & 20 % test

# In[86]:


X=MinMaxScaler().fit_transform(new_data[['cosine_similarity','cwc_min','cwc_max','last_word_eq','first_word_eq','word_share','q1_freq','q2_freq']])#features
Y=new_data["is_duplicate"].values#label

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=10)
print(X.shape)
print(Y.shape)


# In[87]:


def train_model(X_train, X_test, y_train, y_test, model):
    # fit the training dataset on the classifier
    model.fit(X_train, y_train)
    # predict the labels on validation dataset
    y_train_predicted = model.predict(X_train)
    prediction = model.predict(X_test)

    train_err = metrics.accuracy_score(y_train, y_train_predicted)
    test_err = metrics.accuracy_score(y_test, prediction)
    
    print('Train subset accuracy ', train_err)
    print('Test subset accuracy' ,test_err)
    cm=metrics.confusion_matrix(y_test, prediction)
    return cm


# In[88]:


#adaboost classifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),algorithm="SAMME",n_estimators=300)
bdt_graph=train_model(X_train, X_test, y_train, y_test, bdt)


# In[89]:


sns.heatmap(bdt_graph,annot=True)


# In[90]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=3,n_estimators=300)
rf_graph=train_model(X_train, X_test, y_train, y_test, rf)


# In[91]:


sns.heatmap(rf_graph,annot=True)


# In[92]:


#XGBClassifier
get_ipython().run_line_magic('pip', 'install --default-timeout=100 xgboost')
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb_graph=train_model(X_train, X_test, y_train, y_test, xgb)


# In[93]:


sns.heatmap(xgb_graph,annot=True)


# In[ ]:





# In[ ]:





# In[ ]:




