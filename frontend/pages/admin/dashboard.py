"""
Admin dashboard page - Optimized with caching
"""
import streamlit as st
import requests
from config import BACKEND_URL

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_statistics():
    """Fetch statistics with caching"""
    response = requests.get(f"{BACKEND_URL}/api/admin/statistics", timeout=30)
    if response.status_code == 200:
        return response.json()
    return None

def show():
    """Show admin dashboard"""
    st.title("Admin Dashboard")
    st.caption("Dataset Statistics")
    
    with st.spinner("Loading statistics..."):
        try:
            stats = fetch_statistics()
            
            if stats:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Movies", f"{stats.get('total_movies', 0):,}")
                with col2:
                    st.metric("Ratings", f"{stats.get('total_ratings', 0):,}")
                with col3:
                    st.metric("Users", f"{stats.get('total_users', 0):,}")
                with col4:
                    st.metric("Avg Rating", f"{stats.get('avg_rating', 0):.2f}")
                
                st.markdown("---")
                
                # Dataset info
                st.markdown("### Dataset Information")
                st.info("""
                This recommendation system uses a large dataset from Kaggle.
                Statistics are sampled for faster loading.
                """)
                
                # Navigation hint
                st.markdown("### Explore More")
                st.info("Use the navigation sidebar to access Visualizations and Model Metrics.")
                
            else:
                st.error("Error loading statistics")
                st.info("Make sure MongoDB is running and database is seeded.")
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. The dataset may be too large.")
        except Exception as e:
            st.error(f"Connection error: {e}")
            st.info("Make sure the backend server is running at " + BACKEND_URL)
