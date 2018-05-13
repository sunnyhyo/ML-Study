# -*- coding: utf-8 -*-
"""
Created on Sun May 13 01:15:44 2018

@author: HS
"""
#%% cwd
import os

print(os.getcwd())
os.chdir("c:/Users/HS/Documents/GitHub/ml-study/ml-100k")

#%% 12-1
import codecs

def read_data(fin, delim):
    info_li=[]
    
    for line in codecs.open(fin,"r",encoding="latin-1"):
        line_items= line.strip().split(delim)
        
        key=int(line_items[0])
        
        if (len(info_li)+1)!=key:
            print('errors at data_id')
            exit(0)
        info_li.append(line_items[1:])
        
    print("rows in %s: %d" %(fin,len(info_li)))
    
    return(info_li)
    
fin_user="u.user"
fin_movie="u.item"
user_info_li=read_data(fin_user,"|")
movie_info_li=read_data(fin_movie,"|")
#%% 12-2
import numpy as np
#사용자수, 영화수 행렬 만들고 모든 요소를 0 으로 만들기 
R=np.zeros((len(user_info_li),len(movie_info_li)), dtype=np.float64)

for line in codecs.open("u.data","r",encoding="latin-1"):
    user,movie,rating,date=line.strip().split("\t")
    user_index=int(user)-1
    movie_index=int(movie)-1
    
    R[user_index,movie_index]=float(rating)

print(R[0,10])
#%% 12-3

from scipy import stats

user_mean_li=[]
for i in range(0,R.shape[0]):
    user_rating=[x for x in R[i] if x>0.0]
    user_mean_li.append(stats.describe(user_rating).mean)

stats.describe(user_mean_li)
#%%
import matplotlib.pyplot as plt
#plt.plot(user_info_li)
#plt.plot(movie_info_li)
#plt.plot(user_mean_li)
#plt.plot(R)
#행렬시각화...?

print(R.shape) #(943, 1682)
print(R.shape[0]) #943
print(R.shape[1]) #1682
print(R[0,2])
print(R[0])
print(R[1])
print(R[2])
#%% 12-4
movie_mean_li=[]
for i in range(0,R.shape[1]):
    R_T=R.T
    movie_rating=[x for x in R_T[i] if x>0.0]
    movie_mean_li.append(stats.describe(movie_rating).mean)
stats.describe(movie_mean_li)
#%% 12-5   ㅠㅠ
import requests
import json

response=requests.get("http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)")

print("imbd url: %s"%(response.url))

imdb_id= response.url.split("/")[-2]

movie_plot_response=requests.get("http://www.omdbapi.com/?i="+imdb_id+"&plot=full&r=json")
print([x for x in movie_plot_response])

json.loads(movie_plot_response.text)['Plot']

#%%
import sklearn
class sklearn.feature_extraction.text.TfidfVectorizer(
        input='content',encoding='utf-8',decode_error='strict',
        strip_accents=None, lowrcase=True,
        prepeocessor=None,
        tokenizer=None, analyzer="word",stop_words=None,
        token_pattern="(?u)\b\w\w+\b", ngram_range=(1,1),
        max_df=1.0, min_df=1, max_feature=None, vocabulary=None,
        binary=False, dtype=<class "numpy.int64">,
        norm='12',
        use_idf=True, smooth_idf=True, sublinear_tf=False)
#%% 12-6

import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
result_lines = []
movie_plot_li = []
movie_title_li = []

# 영화 ID 1부터 100까지의 영화를 가져옵니다.
# [예제 12-1]에서 생성해둔 무비렌즈 영화 정보 리스트 movie_info_li를 이용합니다.
for movie_info in movie_info_li[:100]:
    movie_url = movie_info[3]
  
    if movie_url == '':
        # 무비렌즈 데이터에 url이 없을 경우의 예외 처리. 타이틀과 플롯은 공백으로 설정합니다.
        print(movie_info)
        movie_title_li.append('')
        movie_plot_li.append('')

    else:
        response = requests.get(movie_url)
        imdb_id = response.url.split('/')[-2]
        # print(imdb_id)
        if imdb_id == 'www.imdb.com':
            print('no imdb id of: %s' % (movie_info[0]))
            # IMDB ID가 없을 경우의 예외 처리
            movie_title = ''
            movie_plot = ''
    
        else:
            try:
                movie_response = requests.get('http://www.omdbapi.com/?i=' + imdb_id + '&plot=full&r=json')
        
            except MissingSchema:
                # OMDB API의 예외 처리
                print('wrong URL: %s' % (movie_info[0]))
                movie_title = ''
                movie_plot = ''

            try:
                movie_title = json.loads(movie_response.text)['Title']
                movie_plot = json.loads(movie_response.text)['Plot']
                #print(movie_response.text)
            except KeyError:
                # API 결과의 예외 처리
                print('incomplete json: %s' % (movie_info[0]))
                movie_title = ''
                movie_plot = ''
        
    result_lines.append("%s\t%s\n" % (movie_title, movie_plot))
    movie_plot_li.append(movie_plot)
    movie_title_li.append(movie_title)
    
print('download complete: %d movie data downloaded'%(len(movie_title_li)))
# 3개 이하의 문서에서 나오는 단어는 TF-IDF 계산에서 제외합니다. 스톱워드는 'english'로 합니다.
vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
X = vectorizer.fit_transform(movie_plot_li)

# TF-IDF로 변환한 키워드의 리스트
# X의 0번 열에 해당하는 키워드가 feature_names[0]의 키워드입니다.
feature_names = vectorizer.get_feature_names()

#%%12-7
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.tokenizer = RegexpTokenizer('(?u)\w\w+')
        # TfidfVectorizer와 같은 방식으로 키워드를 가져옵니다.
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return([self.wnl.lemmatize(t) for t in self.tokenizer.tokenize(doc)])
     
# 사이킷런에 위에서 정의한 토크나이저를 입력으로 넣습니다.
vectorizer2 =TfidfVectorizer(min_df=1,tokenizer=LemmaTokenizer(),stop_words='english')
X = vectorizer2.fit_transform(movie_plot_li)
feature_names = vectorizer2.get_feature_names()
#%%12-8
from sklearn.metrics.pairwise import cosine_similarity

movie_sim = cosine_similarity(X)
def similar_recommend_by_movie_id(movielens_id):
      movie_index = movielens_id-1
      # enumerate 함수로 [(리스트 인덱스 0, 유사도 0), (리스트 인덱스 1, 유사도 1)...]의
      # 리스트를 만듭니다. 그 후 각 튜플의 두 번째 항목, 즉 유사도를 이용하여 내림차순 정렬합니다.
      # 이렇게 만든 리스트의 가장 앞 튜플의 첫 번째 항목이 영화 ID가 됩니다.
      similar_movies = sorted(list(enumerate(movie_sim[movie_index])),key=lambda x:x[1], reverse=True)
      recommended=1
      print("-----recommendation for movie %d------"%(movie))
      for movie_info in similar_movies[1:7]:
            # 주어진 영화와 가장 비슷한 영화는 그 영화 자신이므로 출력 시 제외합니다.
            movie_title= movie_info_li[movie_info[0]]
            print('rank %d recommendation:%s'%(recommended,movie_title[0]))
            recommended+=1
#%%12-9 
from sklearn.metrics import mean_squared_error
import numpy as np
def compute_ALS(R, n_iter, lambda_, k):
    '''임의의 사용자 요인 행렬 X와 임의의 영화 요인 행렬 Y를 생성한 뒤
    교대 최소제곱법을 이용하여 유틸리티 행렬 R을 근사합니다.
    R(ndarray) : 유틸리티 행렬
    lambda_(float) : 정규화 파라미터입니다.
    n_iter(fint) : X와 Y의 갱신 횟수입니다.
    '''
    m, n =R.shape
    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)

    # 각 갱신 때마다 계산한 에러를 저장합니다.
    errors =[]
    for i in range(0, n_iter):
        # [식 6-4]를 구현했습니다.
        # 넘파이의 eye 함수는 파라미터 a를 받아 a x a 크기의 단위행렬을 만듭니다.
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(k), np.dot(Y, R.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k), np.dot(X.T, R))
        
        errors.append(mean_squared_error(R, np.dot(X, Y)))
        
        if i % 10 == 0:
            print('iteration %d is completed'%(i))
            #print(mean_squared_error(R, np.dot(X, Y)))
        
    R_hat = np.dot(X, Y)
    print('Error of rated movies: %.5f'%(mean_squared_error(R, np.dot(X, Y))))
    return(R_hat, errors)
#%%12-10
W = R>0.0
W[W == True] = 1
W[W == False] = 0
W = W.astype(np.float64, copy=False)

def compute_wALS(R,W, n_iter, lambda_, k):
    m,n = R.shape
    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)
    weighted_errors = []
    
    # [예제 12-9]와 달리 가중치 행렬을 넣어서 계산합니다.
    for ii in range(n_iter):
        # 각 사용자와 영화의 가중치 행렬을 이용하여 X와 Y를 갱신합니다.
        for u, Wu in enumerate(W):
            X[u,:] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) +lambda_ * np.eye(k), np.dot(Y, np.dot(np.diag(Wu),R[u,:].T))).T
        for i, Wi in enumerate(W.T):
            Y[:, i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(k), np.dot(X.T, np.dot(np.diag(Wi), R[:, i])))

        # 가중치 행렬을 mean_squared_error 함수의 인자로 사용합니다.
        weighted_errors.append(mean_squared_error(R, np.dot(X, Y),sample_weight=W))
        if ii % 10 == 0:
            print('iteration %d is completed'%(ii))
    
    R_hat = np.dot(X, Y)
    print('Error of rated movies: %.5f'%(mean_squared_error(R, np.dot(X, Y), sample_weight=W)))
    return(R_hat, errors)
#%%12-11
    def compute_GD(R,n_iter, lambda_, learning_rate, k):
    m,n =R.shape
    errors=[]
        
    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)
    
    # 입력받은 반복 횟수만큼 갱신을 반복합니다.
    for ii in range(n_iter):
        for u in range(m):
            for i in range(n):
                if R[u,i]>0:
                    # 새로 정의된 갱신식. 각 사용자 및 상품의 행렬에 대해 하나씩 계산합니다.
                    e_ui = R[u,i]-np.dot(X[u, :], Y[:,i])

                    X[u,:] = X[u,:] + learning_rate * (e_ui* Y[:,i] - lambda_ * X[u,:])
                    Y[:,i] = Y[:,i] + learning_rate * (e_ui * X[u,:] - lambda_ * Y[:,i])  
                    
        errors.append(mean_squared_error(R, np.dot(X, Y)))
        
        if ii % 10 == 0:
            print('iteration %d is completed'%(ii))

    R_hat = np.dot(X, Y)
    print('Error of rated movies: %.5f'%(mean_squared_error(R, R_hat)))

    return(R_hat, errors)

#%%12-12# 예제 12-12 
def train_test_split(R, n_test):
    train = R.copy()
    # 모든 항이 0으로 채워진 학습용 별점 행렬을 만듭니다.
    test = np.zeros(R.shape)
    
    for user in range(R.shape[0]):
        # 각 시용자마다 n_test개의 0이 아닌 항(사용자가 입력한 별점)을 임의로 골라
        # 인덱스를 기억합니다.
        test_index = np.random.choice(R[user, :].nonzero()[0], size=n_test,replace=False)
        
        # 위에서 정한 인덱스에 해당하는 별점을 0으로 만듭니다.
        train[user, test_index] = 0
        
        # 평가 데이터 행렬의 해당 인덱스에 사용자가 입력한 실제 별점을 입력합니다.
        test[user, test_index] = R[user, test_index]
    return(train, test)

# 예제 12-13
def get_test_mse(true,pred):
    # 학습-평가 데이터에서 0이 아닌 값만 이용해서 에러를 계산합니다.
    # true가 평가 데이터, pred가 학습 데이터입니다.
    # 평가 데이터가 0이 아닌 항들의 인덱스에 해당하는 점수만 추출합니다.
    pred = pred[true.nonzero()].flatten()
    true = true[true.nonzero()].flatten()
    return mean_squared_error(true,pred)

#%%12-13
from sklearn.metrics import mean_squared_error
import numpy as np

def compute_ALS2(R, test, n_iter, lambda_, k):
    '''임의의 사용자 요인 행렬 X와 임의의 영화 요인 행렬 Y를 생성하고 교대 최소제곱법을 이용하여
    유틸리티 행렬 R을 근사합니다. 그후 test행렬을 이용하여 평가합니다.
    R(ndarray) : 유틸리티 행렬
    test: 평가행렬
    lambda_(float) : 정규화 파라미터
    n_iter(fint) : X와 Y의 갱신 횟수
    '''
    m,n =R.shape
    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)
    errors =[]
    # 갱신 시마다 계산한 에러를 저장합니다.
    for i in range(0, n_iter):
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(k),np.dot(Y, R.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k), np.dot(X.T, R))
        errors.append(get_test_mse(test,np.dot(X, Y)))

        if i % 10 == 0:
            print('iteration %d is completed'%(i))
    
    R_hat = np.dot(X, Y)
    print('Error of rated movies: %.5f'%(get_test_mse(test,R_hat)))
    return(R_hat, errors)
#%%12-14
from matplotlib import pyplot as plt

plt.xlim(0,20) # x축의 표시 범위를 0-20까지 설정(20은 반복 횟수입니다)
plt.ylim(0,15) # y축의 표시 범위를 0-15까지 설정
plt.xlabel('iteration')
plt.ylabel('MSE')
plt.xticks(x, range(0,20)) # x축에 표시할 숫자를 0부터 19까지의 정수로 함

# 평가 에러를 점선으로 표시
test_plot, = plt.plot(x,test_errors, '--', label='test_error')
# 학습 에러를 실선으로 표시
train_plot, = plt.plot(x,train_errors, label='train_error')

plt.legend(handles=[train_plot, test_plot]) # 범례 생성
plt.show()
#%%12-15
# 근사 행렬의 가장 작은 값을 0으로 만들고자 전체 항의 값에서 작은 값을 뺍니다.
R_hat -= np.min(R_hat)

# 근사 행렬의 가장 큰 값을 5로 만들고자 5를 가장 큰 예측값(np.max(R_hat))으로 나눈 값을 곱합니다.
# 예를 들어 가장 큰 예측값이 3일 경우 3을 5로 만들기 위해서는 5/3을 곱하면 됩니다.
# 위에서 구한 값을 예측 행렬의 모든 항에 곱합니다.
R_hat *= float(5) / np.max(R_hat)

def recommend_by_user(user):
    # 사용자의 ID를 입력으로 받아 그 사용자가 보지 않은 영화를 추천합니다.
    user_index = user-1
    user_seen_movies = sorted(list(enumerate(R_hat[user_index])),
    key=lambda x:x[1], reverse=True)
    recommended=1
    print("-----recommendation for user %d------"%(user))
    for movie_info in user_seen_movies:
        if W[u][movie_info[0]]==0:
            movie_title= movie_info_dic[str(movie_info[0]+1)]
            movie_score= movie_info[1]
            print("rank %d recommendation:%s(%.3f)"%(recommended,movie_title[0], movie_score))
        recommended+=1
        if recommended==6:
            break
 









