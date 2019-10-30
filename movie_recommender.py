import pandas as pd
#we will import pandas for reading csv file
import numpy as np


#used to to find how many times a particular word in occur in a text document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File

df=pd.read_csv('movie_dataset.csv')
print(df.columns)

##Step 2: Select Features
features=['keywords','cast','genres','director']
##Step 3: Create a column in DF which combines all selected features

def combine_features(row):
	return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

for feature in features:
	df[feature]=df[feature].fillna('')

df['combined_features']=df.apply(combine_features,axis=1)


##Step 4: Create count matrix from this new combined column

cv=CountVectorizer()

count_matrix=cv.fit_transform(df['combined_features'])



##Step 5: Compute the Cosine Similarity based on the count_matrix

cosine_sim=cosine_similarity(count_matrix)


'''##### you have to give the name of the movie ##### '''

movie_user_likes = "X-Men"

## Step 6: Get index of this movie from its title

movie_index=get_index_from_title(movie_user_likes)

simliar_movies=list(enumerate(cosine_sim[movie_index]))


## Step 7: Get a list of similar movies in descending order of similarity score

sorted_similar_movies=sorted(simliar_movies,key=lambda x:x[1],reverse=True)


## Step 8: Print titles of first 50 movies
i=0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i=i+1
	if i>10:
		break



