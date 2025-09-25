"""
Main Streamlit application for Rabies Prediction Dashboard
"""

import streamlit as st
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pages import prediction, visualization


def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title="Rabies Prediction System",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App title and description
    st.title("🦠 Rabies Prediction System")
    st.markdown("""
    This dashboard provides machine learning-based predictions for rabies outbreaks 
    and comprehensive data analysis tools.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Model status indicator
    st.sidebar.subheader("System Status")
    
    try:
        # Test model loading
        from utils.model_handler import ModelHandler
        model_handler = ModelHandler()
        
        if model_handler.models is not None:
            st.sidebar.success("✅ Models Loaded")
            model_info = model_handler.get_model_info()
            
            if "error" not in model_info:
                st.sidebar.markdown(f"**Region Classes:** {model_info['region_model']['n_classes']}")
                st.sidebar.markdown(f"**Month Classes:** {model_info['month_model']['n_classes']}")
        else:
            st.sidebar.error("❌ Models Not Loaded")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
    
    st.sidebar.markdown("---")
    
    # Dataset info
    try:
        from utils.data_loader import DataLoader
        data_loader = DataLoader()
        
        if data_loader.preprocessor is not None:
            st.sidebar.success("✅ Preprocessor Loaded")
        else:
            st.sidebar.error("❌ Preprocessor Not Loaded")
            
    except Exception as e:
        st.sidebar.error(f"❌ Data Error: {e}")
    
    # Main content - tabs
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Data Visualization"])
    
    with tab1:
        prediction.show_prediction_page()
    
    with tab2:
        visualization.show_visualization_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Rabies Prediction System | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
