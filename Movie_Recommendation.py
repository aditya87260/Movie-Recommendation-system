import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
warnings.filterwarnings('ignore')
columns_name=["user_id","item_id","rating","timestamp"]
df=pd.read_csv("ml-100k/u.data",sep='\t',names=columns_name)
#print(df.head())
#print(df.shape)
#print(df["user_id"].nunique())
#print(df["item_id"].nunique())
movie_title=pd.read_csv("ml-100k/u.item",sep='\|',header=None)
#print(movie_title.head())
#print(movie_title.shape)
movie_title=movie_title[[0,1]]
movie_title.columns=["item_id","title"]
#print(movie_title.head())
df=pd.merge(df,movie_title,on="item_id")
#print(df.tail())
rating=df.groupby('title').mean()['rating'].sort_values(ascending=False).head()
#print(rating)
count=df.groupby('title').count()['rating'].sort_values(ascending=False)
#print(count.head())
ratings_df=pd.DataFrame(df.groupby('title').mean()['rating'])
#print(ratings_df.head())
ratings_df["num of ratings"]=df.groupby('title').count()['rating']
#print(ratings_df.head())
plt.figure(figsize=(10,6))
plt.hist(ratings_df['num of ratings'],bins=70)
#plt.show()
plt.hist(ratings_df['rating'],bins=70)
#plt.show()
sns.jointplot(x='rating',y='num of ratings',data=ratings_df,alpha=0.5)
#plt.show()
#print(df.head())
movie_pivot=df.pivot_table(index="user_id",columns="title",values="rating")
#print(movie_pivot)
#print(ratings_df.sort_values("num of ratings",ascending=False))
starwars_user_rating=movie_pivot["Star Wars (1977)"]
#print(starwars_user_rating.head())
similar_to_starwars=movie_pivot.corrwith(starwars_user_rating)
corr_starwars=pd.DataFrame(similar_to_starwars,columns=["Correlation"])
corr_starwars.dropna(inplace=True)
#print(corr_starwars)
#print(corr_starwars.sort_values("Correlation",ascending=False).head(10))
corr_starwars=corr_starwars.join(ratings_df["num of ratings"])
#print(corr_starwars.head())
#print(corr_starwars[ratings_df["num of ratings"]>100].sort_values("Correlation",ascending=False))
#Predict Function
def predict_movies(movie_name):
    movie_user_ratings=movie_pivot[movie_name]
    similar_to_movie=movie_pivot.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=["Correlation"])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings_df["num of ratings"])
    predictions=corr_movie[corr_movie["num of ratings"]>100].sort_values("Correlation",ascending=False)
    return predictions
predictions=predict_movies("Titanic (1997)")
print(predictions.head())

