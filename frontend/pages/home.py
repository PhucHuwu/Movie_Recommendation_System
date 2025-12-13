"""
Home page with recommendations
"""
import streamlit as st
import requests
from config import BACKEND_URL

def show():
    """Show home page with recommendations"""
    st.title("Personalized Recommendations")
    
    # Model selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Welcome, User {st.session_state.user_id}!")
    with col2:
        model = st.selectbox(
            "Model",
            ["hybrid", "user_based", "item_based", "neural_cf"],
            format_func=lambda x: {
                "hybrid": "Hybrid (Best)",
                "user_based": "User-Based CF",
                "item_based": "Item-Based CF",
                "neural_cf": "Neural CF"
            }[x]
        )
    
    # Number of recommendations
    num_recs = st.slider("Number of recommendations", 5, 30, 10)
    
    # Get recommendations
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/recommendations/{st.session_state.user_id}",
            params={"model": model, "k": num_recs}
        )
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data['recommendations']
            
            if not recommendations:
                st.warning("No recommendations available.")
                return
            
            # Display recommendations
            st.markdown("---")
            
            # Grid layout
            cols_per_row = 3
            for i in range(0, len(recommendations), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(recommendations):
                        movie = recommendations[idx]
                        
                        with col:
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; height: 200px;">
                                <h4>#{idx+1} {movie['title']}</h4>
                                <p><b>Genres:</b> {movie['genres']}</p>
                                <p style="color: #ff6b6b;"><b>Predicted Rating:</b> {movie['predicted_rating']:.2f}/5.0</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("")
            
        else:
            st.error(f"Error fetching recommendations: {response.json()['detail']}")
            
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.info("Make sure the backend server is running.")
