from flask import Flask,render_template,request,send_from_directory
import pickle
import numpy as np


# Define absolute paths to the pickle files
popular_path = r'C:\Users\mohdy\Desktop\book-recommender-system\popular.pkl'
pt_path = r'C:\Users\mohdy\Desktop\book-recommender-system\pt.pkl'
books_path = r'C:\Users\mohdy\Desktop\book-recommender-system\books.pkl'
similarity_scores_path = r'C:\Users\mohdy\Desktop\book-recommender-system\similarity_scores.pkl'

# Load the pickle files
popular_df = pickle.load(open(popular_path, 'rb'))
pt = pickle.load(open(pt_path, 'rb'))
books = pickle.load(open(books_path, 'rb'))
similarity_scores = pickle.load(open(similarity_scores_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
@app.route('/contact')
def contact_us():
    return render_template('contact.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    # Check if the user_input exists in pt.index
    if user_input not in pt.index:
        error_message = "Search valid names only"
        return render_template('recommend.html', error_message=error_message)

    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)