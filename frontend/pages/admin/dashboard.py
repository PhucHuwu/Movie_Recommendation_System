"""
Admin dashboard page
"""
import streamlit as st
import requests
from config import BACKEND_URL

def show():
    """Show admin dashboard"""
    st.title("Admin Dashboard")
    st.caption("Dataset Statistics (Accessible to all users)")
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/admin/statistics")
        
        if response.status_code == 200:
            stats = response.json()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Movies", f"{stats['total_movies']:,}")
            with col2:
                st.metric("Total Ratings", f"{stats['total_ratings']:,}")
            with col3:
                st.metric("Total Users", f"{stats['total_users']:,}")
            with col4:
                st.metric("Average Rating", f"{stats['avg_rating']:.2f}")
            
            st.markdown("---")
            
            # Top genres
            st.markdown("### Top Genres")
            
            if stats.get('top_genres'):
                # Create two columns for genre display
                col1, col2 = st.columns(2)
                
                genres = stats['top_genres']
                mid = len(genres) // 2
                
                with col1:
                    for genre, count in genres[:mid]:
                        st.markdown(f"**{genre}:** {count:,} movies")
                
                with col2:
                    for genre, count in genres[mid:]:
                        st.markdown(f"**{genre}:** {count:,} movies")
            
            st.markdown("---")
            
            # Dataset info
            st.markdown("### Dataset Information")
            st.info("""
            This recommendation system uses a curated dataset from Kaggle.
            The dataset contains movie information and user ratings, enabling 
            personalized recommendations through multiple ML algorithms.
            """)
            
        else:
            st.error("Error loading statistics")
            
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.info("Make sure the backend server is running.")
