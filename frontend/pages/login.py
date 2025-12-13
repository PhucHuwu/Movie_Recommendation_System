"""
Login page
"""
import streamlit as st
import requests
from config import BACKEND_URL

def show():
    """Show login page"""
    st.title("Movie Recommendation System")
    st.markdown("### Login to Get Personalized Recommendations")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        
        # Login form
        with st.form("login_form"):
            user_id = st.number_input(
                "Enter your User ID",
                min_value=1,
                step=1,
                help="Enter a valid user ID from the dataset"
            )
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                try:
                    # Call backend API
                    response = requests.post(
                        f"{BACKEND_URL}/api/auth/login",
                        json={"userId": int(user_id)}
                    )
                    
                    if response.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.user_id = int(user_id)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {response.json()['detail']}")
                        
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    st.info("Make sure the backend server is running at " + BACKEND_URL)
        
        st.markdown("---")
        st.info("This system uses a curated dataset. You can only login with existing user IDs.")
        st.caption("Note: New user registration is not available to preserve dataset integrity.")
