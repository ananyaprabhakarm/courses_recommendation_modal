import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import uvicorn
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('DataSet.csv')

app = FastAPI()

new_df = df.copy()

new_df['Rating'].fillna(new_df['Rating'].mean(), inplace=True)
new_df['Review Count'].fillna(new_df['Review Count'].mean(), inplace=True)
new_df['Prerequisites'].fillna("No Prerequisites Required", inplace=True)
new_df['Affiliates'].fillna("None", inplace=True)
new_df['Level'].fillna("beginner", inplace=True)

new_df.isnull().sum()

new_df_2 = new_df[['Title','Level','Prerequisites','Skills Covered']]
new_df_2.head()

vectorizer = TfidfVectorizer()

df_vector = vectorizer.fit_transform(new_df_2['Skills Covered'])

similarity = linear_kernel(df_vector, df_vector)

# vectorizer = TfidfVectorizer(stop_words='english')


def recommend(user_preferences, new_df_2):
#   new_df_2['features'] = new_df_2['Title'] + ' ' +  new_df_2['Level'] + ' ' +  new_df_2['Prerequisites'] + ' ' + new_df_2['Skills Covered']

  course_matrix = vectorizer.transform([user_preferences])
#   print(course_matrix.shape)
#   print(similarity.shape)
  cosine_similarities = linear_kernel(course_matrix, vectorizer.transform(new_df_2['Skills Covered']))
#   user_course_index = new_df_2[new_df_2['Title'] == user_preferences]

  similarity_scores = list(enumerate(cosine_similarities[0]))
  sorted_courses = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
  top_n_recommendations = [(new_df_2.iloc[idx]['Title'], score) for idx, score in sorted_courses[1:6]]
  return top_n_recommendations

# print(recommend('Python', new_df_2))  



@app.get("/recommendations/{user_preferences}")
async def read_recommendations(user_preferences: str):
    recommendations = recommend(user_preferences, new_df_2)

    # formatted_recommendations = [{"course": course, "similarity_score": score} for course, score in recommendations]

    return {"user_preferences": user_preferences, "recommendations":recommendations}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
