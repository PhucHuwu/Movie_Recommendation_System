"""
Search page - Optimized with caching and pagination
"""
import streamlit as st
import requests
from config import BACKEND_URL

@st.cache_data(ttl=300)
def fetch_top_movies():
    """Fetch top movies with caching"""
    response = requests.get(f"{BACKEND_URL}/api/movies/top/rated", params={"limit": 20}, timeout=30)
    if response.status_code == 200:
        return response.json()
    return None

def show():
    """Show search page"""
    st.title("Search Movies")
    
    # Search bar
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("Search by title or genre", placeholder="e.g., Inception, Action, Comedy")
    with col2:
        page = st.number_input("Page", min_value=1, value=1, step=1)
    
    if search_query:
        with st.spinner("Searching..."):
            try:
                response = requests.get(
                    f"{BACKEND_URL}/api/movies/search",
                    params={"q": search_query, "limit": 20, "page": page},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data['results']
                    
                    st.markdown(f"**Found {data.get('total', data['count'])} movies** (Page {data.get('page', 1)}/{data.get('total_pages', 1)})")
                    
                    if not results:
                        st.info("No movies found matching your search.")
                        return
                    
                    # Display results in grid
                    for movie in results:
                        with st.expander(f"**{movie['title']}** ({movie.get('year', 'N/A')})"):
                            st.markdown(f"**Genres:** {movie['genres']}")
                            st.markdown(f"**Movie ID:** {movie['movieId']}")
                
                else:
                    st.error("Search failed")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out")
            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        # Show top movies
        st.markdown("### Top Rated Movies")
        
        with st.spinner("Loading top movies..."):
            data = fetch_top_movies()
            
            if data and data.get('movies'):
                movies = data['movies']
                
                for i, movie in enumerate(movies, 1):
                    avg = movie.get('avgRating', 0)
                    num = movie.get('numRatings', 0)
                    with st.expander(f"#{i} **{movie['title']}** - {avg:.2f} ({num} ratings)"):
                        st.markdown(f"**Genres:** {movie['genres']}")
                        if movie.get('year'):
                            st.markdown(f"**Year:** {movie['year']}")
                
                if data.get('cached'):
                    st.caption("Data from cache")
            else:
                st.info("Unable to load top movies. Make sure MongoDB is running.")
        
        # Refresh button
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
