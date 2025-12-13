"""
Data visualization page - Optimized with caching and sampling
"""
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from config import BACKEND_URL

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_rating_distribution():
    """Fetch rating distribution with caching"""
    response = requests.get(f"{BACKEND_URL}/api/admin/visualizations/rating-distribution", timeout=30)
    if response.status_code == 200:
        return response.json()
    return None

@st.cache_data(ttl=300)
def fetch_genre_distribution():
    """Fetch genre distribution with caching"""
    response = requests.get(f"{BACKEND_URL}/api/admin/visualizations/genre-distribution", timeout=30)
    if response.status_code == 200:
        return response.json()
    return None

def show():
    """Show visualization page"""
    st.title("Data Visualizations")
    st.caption("Visual analysis of the dataset (sampled for performance)")
    
    # Tabs for different visualizations
    tab1, tab2 = st.tabs(["Rating Distribution", "Genre Distribution"])
    
    with tab1:
        st.subheader("Rating Distribution")
        with st.spinner("Loading..."):
            data = fetch_rating_distribution()
            if data and data.get('labels'):
                fig = px.bar(
                    x=data['labels'],
                    y=data['values'],
                    labels={'x': 'Rating', 'y': 'Count'},
                    title='Distribution of Ratings'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                if data.get('note'):
                    st.caption(data['note'])
            else:
                st.warning("No rating data available")
    
    with tab2:
        st.subheader("Genre Distribution")
        with st.spinner("Loading..."):
            data = fetch_genre_distribution()
            if data and data.get('labels'):
                fig = px.bar(
                    x=data['values'],
                    y=data['labels'],
                    orientation='h',
                    labels={'x': 'Movie Count', 'y': 'Genre'},
                    title='Top 15 Genres'
                )
                fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No genre data available")
    
    # Clear cache button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
