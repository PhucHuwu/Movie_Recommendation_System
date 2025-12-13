"""
Search page
"""
import streamlit as st
import requests
from config import BACKEND_URL

def show():
    """Show search page"""
    st.title("🔍 Search Movies")
    
    # Search bar
    search_query = st.text_input("Search by title or genre", placeholder="e.g., Inception, Action, Comedy")
    
    if search_query:
        try:
            response = requests.get(
                f"{BACKEND_URL}/api/movies/search",
                params={"q": search_query, "limit": 50}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data['results']
                
                st.markdown(f"### Found {data['count']} movies")
                
                if not results:
                    st.info("No movies found matching your search.")
                    return
                
                # Display results
                for movie in results:
                    with st.expander(f"**{movie['title']}**"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Genres:** {movie['genres']}")
                            st.markdown(f"**Movie ID:** {movie['movieId']}")
                            if movie.get('year'):
                                st.markdown(f"**Year:** {movie['year']}")
                        
                        with col2:
                            if st.button("View Details", key=f"view_{movie['movieId']}"):
                                show_movie_details(movie['movieId'])
            
            else:
                st.error("Search failed")
                
        except Exception as e:
            st.error(f"Connection error: {e}")
    else:
        # Show top movies
        st.markdown("### 🔥 Top Rated Movies")
        
        try:
            response = requests.get(f"{BACKEND_URL}/api/movies/top/rated", params={"limit": 20})
            
            if response.status_code == 200:
                data = response.json()
                movies = data['movies']
                
                for i, movie in enumerate(movies, 1):
                    with st.expander(f"#{i} **{movie['title']}** - ⭐ {movie['avgRating']:.2f} ({movie['numRatings']} ratings)"):
                        st.markdown(f"**Genres:** {movie['genres']}")
                        if movie.get('year'):
                            st.markdown(f"**Year:** {movie['year']}")
            
        except Exception as e:
            st.error(f"Error loading top movies: {e}")

def show_movie_details(movie_id):
    """Show detailed movie information"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/movies/{movie_id}")
        
        if response.status_code == 200:
            movie = response.json()
            
            st.markdown("---")
            st.markdown(f"## {movie['title']}")
            st.markdown(f"**Genres:** {movie['genres']}")
            if movie.get('year'):
                st.markdown(f"**Year:** {movie['year']}")
            
            # Similar movies
            if movie.get('similar_movies'):
                st.markdown("### 🎬 Similar Movies")
                for sim_movie in movie['similar_movies']:
                    st.markdown(f"- **{sim_movie['title']}** (Similarity: {sim_movie['similarity_score']:.2f})")
    
    except Exception as e:
        st.error(f"Error loading movie details: {e}")
