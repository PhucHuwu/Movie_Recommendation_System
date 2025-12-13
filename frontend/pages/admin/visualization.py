"""
Data visualization page
"""
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from config import BACKEND_URL

def show():
    """Show data visualizations"""
    st.title("Data Visualizations")
    
    try:
        # Rating Distribution
        st.markdown("### Rating Distribution")
        rating_response = requests.get(f"{BACKEND_URL}/api/admin/visualizations/rating-distribution")
        
        if rating_response.status_code == 200:
            rating_data = rating_response.json()
            
            fig = px.bar(
                x=rating_data['labels'],
                y=rating_data['values'],
                labels={'x': 'Rating', 'y': 'Count'},
                title='Distribution of Movie Ratings'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Genre Distribution
        st.markdown("### Genre Distribution")
        genre_response = requests.get(f"{BACKEND_URL}/api/admin/visualizations/genre-distribution")
        
        if genre_response.status_code == 200:
            genre_data = genre_response.json()
            
            fig = px.bar(
                x=genre_data['labels'],
                y=genre_data['values'],
                labels={'x': 'Genre', 'y': 'Number of Movies'},
                title='Top 15 Movie Genres'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # User Activity
        st.markdown("### User Activity Distribution")
        activity_response = requests.get(f"{BACKEND_URL}/api/admin/visualizations/user-activity")
        
        if activity_response.status_code == 200:
            activity_data = activity_response.json()
            buckets = activity_data['buckets']
            
            if buckets:
                # Create labels
                labels = []
                values = []
                
                for bucket in buckets:
                    bucket_id = bucket['_id']
                    if isinstance(bucket_id, (int, float)):
                        labels.append(f"{int(bucket_id)}+")
                    else:
                        labels.append(str(bucket_id))
                    values.append(bucket['users'])
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title='Number of Ratings per User'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.info("These visualizations help understand the dataset characteristics and user behavior patterns.")
        
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.info("Make sure the backend server is running.")
