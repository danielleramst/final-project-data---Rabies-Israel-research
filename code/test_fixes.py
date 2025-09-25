#!/usr/bin/env python3
"""
Test script to verify the fixes for the Rabies Dashboard
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_model_handler():
    """Test the ModelHandler with feature selection"""
    print("🧪 Testing ModelHandler...")
    
    try:
        from utils.model_handler import ModelHandler
        model_handler = ModelHandler()
        
        if model_handler.models is not None:
            print("✅ Models loaded successfully")
            
            # Test feature availability
            features = model_handler.get_available_features()
            print(f"✅ Available features: {len(features)}")
            print(f"   Sample features: {features[:5] if features else 'None'}")
            
            return True
        else:
            print("❌ Models not loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error testing ModelHandler: {e}")
        return False

def test_data_loader():
    """Test the DataLoader"""
    print("\n🧪 Testing DataLoader...")
    
    try:
        from utils.data_loader import DataLoader
        data_loader = DataLoader()
        
        if data_loader.preprocessor is not None:
            print("✅ DataLoader initialized successfully")
            
            # Test loading existing data
            df = data_loader.load_excel_file("data/Rabies__Weather__War_Combined_1.4.25.xlsx")
            if not df.empty:
                print(f"✅ Data loaded: {df.shape}")
                return True
            else:
                print("❌ No data loaded")
                return False
        else:
            print("❌ Preprocessor not loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error testing DataLoader: {e}")
        return False

def test_visualizer():
    """Test the Visualizer"""
    print("\n🧪 Testing Visualizer...")
    
    try:
        from utils.visualizer import Visualizer
        visualizer = Visualizer()
        print("✅ Visualizer initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Visualizer: {e}")
        return False

def test_prediction_with_features():
    """Test prediction with feature selection"""
    print("\n🧪 Testing Prediction with Feature Selection...")
    
    try:
        from utils.model_handler import ModelHandler
        from utils.data_loader import DataLoader
        
        model_handler = ModelHandler()
        data_loader = DataLoader()
        
        if model_handler.models is None or data_loader.preprocessor is None:
            print("❌ Models or preprocessor not available")
            return False
        
        # Load sample data
        df = data_loader.load_excel_file("data/Rabies__Weather__War_Combined_1.4.25.xlsx")
        if df.empty:
            print("❌ No sample data available")
            return False
        
        # Get a sample
        sample = df.head(1)
        processed_sample = data_loader.preprocess_data(sample)
        
        if processed_sample.empty:
            print("❌ Failed to process sample")
            return False
        
        # Test with all features
        result_all = model_handler.predict_both(processed_sample)
        
        if "error" in result_all.get("region", {}) or "error" in result_all.get("month", {}):
            print(f"❌ Prediction with all features failed:")
            print(f"   Region: {result_all.get('region', {}).get('error', 'OK')}")
            print(f"   Month: {result_all.get('month', {}).get('error', 'OK')}")
            return False
        
        print("✅ Prediction with all features successful")
        
        # Test with selected features
        available_features = model_handler.get_available_features()
        if available_features and len(available_features) > 3:
            selected_features = available_features[:3]  # Use first 3 features
            result_selected = model_handler.predict_both(processed_sample, selected_features)
            
            if "error" in result_selected.get("region", {}) or "error" in result_selected.get("month", {}):
                print(f"❌ Prediction with selected features failed:")
                print(f"   Region: {result_selected.get('region', {}).get('error', 'OK')}")
                print(f"   Month: {result_selected.get('month', {}).get('error', 'OK')}")
                return False
            
            print(f"✅ Prediction with {len(selected_features)} selected features successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing prediction with features: {e}")
        return False

def main():
    """Run all tests"""
    print("🦠 Testing Rabies Dashboard Fixes\n")
    
    tests = [
        test_model_handler,
        test_data_loader,
        test_visualizer,
        test_prediction_with_features
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Dashboard should work correctly.")
        print("\n🚀 To run the dashboard:")
        print("   ./run_dashboard.sh")
        print("   or")
        print("   streamlit run app/main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
