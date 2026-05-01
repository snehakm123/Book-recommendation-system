import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import logging

# Set page config
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 3rem !important;
    }
    .book-container {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    .book-container:hover {
        transform: translateY(-5px);
    }
    .welcome-text {
        font-size: 1.2rem;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Set up logging to Streamlit
logging.basicConfig(level=logging.INFO)

# Load data
@st.cache_data
def load_data():
    books_df = pd.read_csv('archive (1)/Books.csv', low_memory=False)
    ratings_df = pd.read_csv('archive (1)/Ratings.csv')
    users_df = pd.read_csv('archive (1)/Users.csv')
    return books_df, ratings_df, users_df

def get_book_recommendations(selected_book, books_df, ratings_df, n_recommendations=5):
    try:
        # Get the ISBN of the selected book
        selected_isbn = books_df[books_df['Book-Title'] == selected_book]['ISBN'].iloc[0]
        
        # Get all ratings for the selected book
        book_ratings = ratings_df[ratings_df['ISBN'] == selected_isbn]
        
        if len(book_ratings) > 0:
            # Find users who rated this book
            users_who_rated = book_ratings['User-ID'].unique()
            
            # Get all ratings by these users
            similar_users_ratings = ratings_df[ratings_df['User-ID'].isin(users_who_rated)]
            
            # Calculate average rating per book
            book_avg_ratings = similar_users_ratings.groupby('ISBN')['Book-Rating'].agg(['mean', 'count']).reset_index()
            
            # Filter books with at least 3 ratings
            book_avg_ratings = book_avg_ratings[book_avg_ratings['count'] >= 3]
            
            # Sort by average rating
            recommended_books = book_avg_ratings.sort_values('mean', ascending=False)
            
            # Remove the selected book from recommendations
            recommended_books = recommended_books[recommended_books['ISBN'] != selected_isbn]
            
            # Get top N recommendations
            recommended_books = recommended_books.head(n_recommendations)
            
            # Get book details for recommended books
            final_recommendations = books_df[books_df['ISBN'].isin(recommended_books['ISBN'])]
            
            if not final_recommendations.empty:
                return final_recommendations
        
        # If collaborative filtering didn't work, use content-based filtering
        st.info("Using title similarity for recommendations...")
        
        # Combine title and author for better recommendations
        books_df['features'] = books_df['Book-Title'].fillna('') + ' ' + books_df['Book-Author'].fillna('')
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(books_df['features'])
        
        # Get the index of the selected book
        idx = books_df[books_df['Book-Title'] == selected_book].index[0]
        
        # Calculate similarity scores
        cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        
        # Get indices of top similar books
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]  # Exclude the book itself
        
        book_indices = [i[0] for i in sim_scores]
        return books_df.iloc[book_indices]
            
    except Exception as e:
        st.error(f"Error in recommendation system: {str(e)}")
        return None

def get_best_cover(book):
    # 1. Use Image-URL-M if valid
    url = book.get('Image-URL-M', '')
    if isinstance(url, str) and url.startswith('http') and not url.endswith('nophoto.gif'):
        logging.info(f"Using dataset image for {book.get('Book-Title', '')}")
        return url
    # 2. Try Open Library by ISBN
    isbn = str(book.get('ISBN', '')).strip()
    if isbn and isbn != 'nan':
        openlib_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
        try:
            resp = requests.get(openlib_url)
            # Open Library returns a default image for missing covers, which is small (<2KB)
            if resp.status_code == 200 and resp.headers['Content-Type'].startswith('image') and len(resp.content) > 2500:
                logging.info(f"Using Open Library cover for {book.get('Book-Title', '')}")
                return openlib_url
        except Exception as e:
            logging.warning(f"Open Library error for {isbn}: {e}")
    # 3. Try Google Books API by title/author
    title = str(book.get('Book-Title', '')).strip()
    author = str(book.get('Book-Author', '')).strip()
    if title:
        try:
            q = f"intitle:{title}"
            if author:
                q += f"+inauthor:{author}"
            api_url = f"https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=1"
            resp = requests.get(api_url)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get('items', [])
                if items:
                    image_links = items[0]['volumeInfo'].get('imageLinks', {})
                    if 'thumbnail' in image_links:
                        logging.info(f"Using Google Books cover for {book.get('Book-Title', '')}")
                        return image_links['thumbnail']
        except Exception as e:
            logging.warning(f"Google Books error for {title}: {e}")
    # 4. Fallback placeholder
    logging.info(f"Using placeholder for {book.get('Book-Title', '')}")
    return "https://via.placeholder.com/150x200?text=No+Image"

try:
    books_df, ratings_df, users_df = load_data()
    
    # Main title
    st.title("📚 Book Recommendation System")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Search Books", "Get Recommendations", "Popular Books"])
    
    if page == "Home":
        st.header("Welcome to the Book Recommendation System!")
        st.markdown('<p class="welcome-text">Discover your next favorite book with our intelligent recommendation system.</p>', unsafe_allow_html=True)
        
        # Display some statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Total Books", f"{len(books_df):,}")
        with col2:
            st.metric("👥 Total Users", f"{len(users_df):,}")
        with col3:
            st.metric("⭐ Total Ratings", f"{len(ratings_df):,}")
            
        # Display recently released books from the internet (Google Books API)
        st.subheader("Recently Released Books")
        def fetch_recent_books_from_google(n=6, from_year=2025):
            # Fetches recent books from Google Books API (published from from_year onwards)
            results = []
            try:
                # Use a standard query and sort by newest, then filter by year locally
                api_url = "https://www.googleapis.com/books/v1/volumes"
                params = {
                    'q': 'subject:fiction',
                    'orderBy': 'newest',
                    'printType': 'books',
                    'maxResults': 40
                }
                resp = requests.get(api_url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    count = 0
                    for item in data.get('items', []):
                        info = item.get('volumeInfo', {})
                        title = info.get('title', 'Unknown Title')
                        authors = ', '.join(info.get('authors', [])) if info.get('authors') else 'Unknown Author'
                        publisher = info.get('publisher', 'Unknown Publisher')
                        pub_date = info.get('publishedDate', '')
                        # Try to extract year (some dates are YYYY-MM-DD or YYYY)
                        try:
                            pub_year = int(pub_date[:4])
                        except Exception:
                            pub_year = 'N/A'
                        image_url = info.get('imageLinks', {}).get('thumbnail', "https://via.placeholder.com/150x200?text=No+Image")
                        # Only include books from from_year onwards
                        if pub_year != 'N/A' and pub_year >= from_year:
                            results.append({
                                'title': title,
                                'authors': authors,
                                'publisher': publisher,
                                'year': pub_year,
                                'image_url': image_url
                            })
                            count += 1
                        if count >= n:
                            break
                else:
                    logging.warning(f"Google Books API returned status {resp.status_code}")
            except Exception as e:
                st.warning(f"Could not fetch recent books from Google Books: {e}")
            return results
        
        recent_internet_books = fetch_recent_books_from_google(6, from_year=2025)
        for book in recent_internet_books:
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(book['image_url'], width=150)
                with col2:
                    st.markdown(f"### {book['title']}")
                    st.write(f"**Author(s):** {book['authors']}")
                    st.write(f"**Publisher:** {book['publisher']}")
                    st.write(f"**Year:** {book['year']}")
    
    elif page == "Search Books":
        st.header("Search Books")
        search_term = st.text_input("Enter book title, author, or ISBN")
        
        if search_term:
            # Search in books dataframe
            results = books_df[
                books_df['Book-Title'].str.contains(search_term, case=False, na=False) |
                books_df['Book-Author'].str.contains(search_term, case=False, na=False) |
                books_df['ISBN'].str.contains(search_term, case=False, na=False)
            ]
            
            if not results.empty:
                st.write(f"Found {len(results)} results:")
                for _, book in results.iterrows():
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(book['Image-URL-M'] if 'Image-URL-M' in book else "https://placeholder.com/150", width=150)
                        with col2:
                            st.markdown(f"### {book['Book-Title']}")
                            st.write(f"**Author:** {book['Book-Author']}")
                            st.write(f"**Publisher:** {book['Publisher']}")
                            st.write(f"**Year:** {book['Year-Of-Publication']}")
            else:
                st.warning("No books found matching your search terms.")
    
    elif page == "Get Recommendations":
        st.header("Get Personalized Recommendations")
        
        # Get unique book titles and handle any potential NaN values
        book_titles = books_df['Book-Title'].dropna().unique().tolist()
        book_titles.sort()  # Sort alphabetically
        
        # Simple recommendation based on book similarity
        selected_book = st.selectbox(
            "Select a book you like:",
            options=book_titles,
            index=None,
            placeholder="Choose a book...",
            key="book_selector"
        )
        
        if selected_book:
            if st.button("Get Recommendations"):
                with st.spinner("Finding recommendations..."):
                    # Get recommendations
                    recommendations = get_book_recommendations(selected_book, books_df, ratings_df)
                    
                    if recommendations is not None and not recommendations.empty:
                        st.subheader("Recommended Books:")
                        for _, book in recommendations.iterrows():
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.image(book['Image-URL-M'] if 'Image-URL-M' in book else "https://placeholder.com/150", width=150)
                                with col2:
                                    st.markdown(f"### {book['Book-Title']}")
                                    st.write(f"**Author:** {book['Book-Author']}")
                                    st.write(f"**Publisher:** {book['Publisher']}")
                                    st.write(f"**Year:** {book['Year-Of-Publication']}")
                    else:
                        st.warning("Could not generate recommendations at this time. Please try another book.")
        else:
            st.info("Please select a book to get recommendations.")
    
    elif page == "Popular Books":
        st.header("Popular Books")
        
        # Calculate average ratings and number of ratings
        book_stats = ratings_df.groupby('ISBN').agg({
            'Book-Rating': ['count', 'mean']
        }).reset_index()
        book_stats.columns = ['ISBN', 'rating_count', 'rating_mean']
        
        # Filter books with minimum number of ratings
        min_ratings = st.slider("Minimum number of ratings", 1, 100, 10)
        popular_books = book_stats[book_stats['rating_count'] >= min_ratings]
        
        # Merge with books dataframe
        popular_books = popular_books.merge(books_df, on='ISBN')
        popular_books = popular_books.sort_values('rating_mean', ascending=False)
        
        # Display top books
        for _, book in popular_books.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(book['Image-URL-M'] if 'Image-URL-M' in book else "https://placeholder.com/150", width=150)
                with col2:
                    st.markdown(f"### {book['Book-Title']}")
                    st.write(f"**Author:** {book['Book-Author']}")
                    st.write(f"**Average Rating:** {book['rating_mean']:.2f} ⭐")
                    st.write(f"**Number of Ratings:** {int(book['rating_count'])}")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please make sure the data files are in the correct location (archive (1) folder) and have the correct format.") 