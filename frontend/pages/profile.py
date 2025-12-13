"""
User profile page
"""
import streamlit as st
import requests
import pandas as pd
from config import BACKEND_URL

def show():
    """Show user profile page"""
    st.title("👤 My Profile")
    
    # Get user info
    try:
        user_response = requests.get(f"{BACKEND_URL}/api/auth/me/{st.session_state.user_id}")
        
        if user_response.status_code == 200:
            user_data = user_response.json()
            
            # User stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("User ID", user_data['userId'])
            with col2:
                st.metric("Total Ratings", user_data['totalRatings'])
            with col3:
                st.metric("Average Rating", f"{user_data['avgRating']:.2f}")
            
            st.markdown("---")
            
            # Get user's rating history
            ratings_response = requests.get(
                f"{BACKEND_URL}/api/recommendations/user/{st.session_state.user_id}/ratings"
            )
            
            if ratings_response.status_code == 200:
                ratings_data = ratings_response.json()
                ratings = ratings_data['ratings']
                
                st.markdown("### 📚 My Rating History")
                
                if not ratings:
                    st.info("No ratings found.")
                    return
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(ratings)
                
                # Select and rename columns
                display_df = df[['movieTitle', 'movieGenres', 'rating', 'timestamp']].copy()
                display_df.columns = ['Movie', 'Genres', 'Rating', 'Timestamp']
                
                # Sort by timestamp (most recent first)
                if 'Timestamp' in display_df.columns:
                    display_df = display_df.sort_values('Timestamp', ascending=False)
                
                # Display as table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Rating distribution
                st.markdown("### 📊 My Rating Distribution")
                rating_counts = df['rating'].value_counts().sort_index()
                st.bar_chart(rating_counts)
                
        else:
            st.error("Error loading user profile")
            
    except Exception as e:
        st.error(f"Connection error: {e}")
