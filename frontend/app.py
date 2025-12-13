"""
Main Streamlit Application
"""
import streamlit as st
import sys
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.title("Movie Recommender")
        
        if st.session_state.logged_in:
            st.success(f"Logged in as User {st.session_state.user_id}")
            
            # Navigation
            st.markdown("---")
            page = st.radio(
                "Navigation",
                ["Home", "Search", "My Profile", "Admin Dashboard", "Data Visualization", "Model Evaluation"]
            )
            
            st.markdown("---")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_id = None
                st.rerun()
        else:
            st.info("Please login to continue")
            page = "Login"
    
    # Main content
    if not st.session_state.logged_in:
        from frontend.pages import login
        login.show()
    else:
        if page == "Home":
            from frontend.pages import home
            home.show()
        elif page == "Search":
            from frontend.pages import search
            search.show()
        elif page == "My Profile":
            from frontend.pages import profile
            profile.show()
        elif page == "Admin Dashboard":
            from frontend.pages.admin import dashboard
            dashboard.show()
        elif page == "Data Visualization":
            from frontend.pages.admin import visualization
            visualization.show()
        elif page == "Model Evaluation":
            from frontend.pages.admin import models
            models.show()

if __name__ == "__main__":
    main()
