import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Define paths to the data files
base_path = r'C:\Users\mohdy\Desktop\book-recommender-system'
books_path = f'{base_path}\\Books.csv'
users_path = f'{base_path}\\Users.csv'
ratings_path = f'{base_path}\\Ratings.csv'

# Load the data
books = pd.read_csv(books_path)
users = pd.read_csv(users_path)
ratings = pd.read_csv(ratings_path)

# Merge ratings with book information
ratings_with_name = ratings.merge(books, on='ISBN')

# Ensure 'Book-Rating' is numeric
ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')

# Drop rows where 'Book-Rating' could not be converted to numeric
ratings_with_name = ratings_with_name.dropna(subset=['Book-Rating'])

# Calculate the number of ratings for each book
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

# Calculate the average rating for each book
avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

# Merge the number of ratings and average ratings
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)

# Merge with books data and select necessary columns
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

# Filter users who have rated more than 200 books
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index

# Filter ratings by these users
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

# Filter books that have at least 50 ratings
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

# Get final ratings for these books
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

# Create the pivot table for book-user matrix
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Calculate similarity scores using cosine similarity
similarity_scores = cosine_similarity(pt)

# Function to recommend books
def recommend(book_name):
    if book_name not in pt.index:
        return f"Book '{book_name}' not found in the dataset."
    
    # Get index of the book in the pivot table
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data

# Save the objects using pickle
pickle.dump(popular_df, open(f'{base_path}\\popular.pkl', 'wb'))
pickle.dump(pt, open(f'{base_path}\\pt.pkl', 'wb'))
pickle.dump(books, open(f'{base_path}\\books.pkl', 'wb'))
pickle.dump(similarity_scores, open(f'{base_path}\\similarity_scores.pkl', 'wb'))
