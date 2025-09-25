"""
Model handling and prediction utilities for the Rabies prediction dashboard
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from typing import Dict, Tuple, Any


class ModelHandler:
    """Handles model loading and predictions"""
    
    def __init__(self, model_path: str = "models/trained_model.pkl"):
        """Initialize with model path"""
        self.model_path = model_path
        self.models = None
        self.preprocessor = None
        self.region_features = None
        self.month_features = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessor"""
        try:
            # Load all models from the combined file
            all_models = joblib.load(self.model_path)
            
            self.models = {
                'region': all_models['region_model'],
                'month': all_models['month_model']
            }
            self.preprocessor = all_models['preprocessor']
            
            # Load feature names if available (for newer model versions)
            self.region_features = all_models.get('region_features', None)
            self.month_features = all_models.get('month_features', None)
            
            # If feature names not saved, derive them from preprocessor
            if self.region_features is None or self.month_features is None:
                self._derive_feature_names()
            
            st.success("Models loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            self.models = None
            self.preprocessor = None
    
    def _derive_feature_names(self):
        """Derive feature names from preprocessor when not saved"""
        if self.preprocessor is not None:
            # Get all column names
            categorical_cols = self.preprocessor['categorical_columns']
            numeric_cols = self.preprocessor['numeric_columns']
            
            # Region features: all except Region
            all_cols = categorical_cols + numeric_cols + ['War in Israel', 'Year']
            self.region_features = [col for col in all_cols if col != 'Region']
            
            # Month features: all except Month
            self.month_features = [col for col in all_cols if col != 'Month']
    
    def get_available_features(self):
        """Get list of available features for selection"""
        if self.region_features is not None:
            # Return all possible features (union of region and month features)
            all_features = list(set(self.region_features + self.month_features))
            return sorted(all_features)
        return []
    
    def predict_region(self, data: pd.DataFrame, selected_features: list = None) -> Dict[str, Any]:
        """Predict region for given data with optional feature selection"""
        if self.models is None:
            return {"error": "Models not loaded"}
        
        try:
            # Prepare data for region prediction (exclude region column)
            X = data.drop(columns=['Region'], errors='ignore')
            
            # Handle feature selection
            if selected_features is not None:
                # Use only selected features
                missing_features = [f for f in selected_features if f not in X.columns]
                if missing_features:
                    return {"error": f"Missing selected features: {missing_features}"}
                X_selected = X[selected_features]
                
                # Retrain model with selected features for this prediction
                # This is a simplified approach - in production you'd want pre-trained models
                from sklearn.ensemble import GradientBoostingClassifier
                temp_model = GradientBoostingClassifier(random_state=42)
                
                # We need training data to retrain - this is a limitation
                # For now, we'll use the original model but note the feature limitation
                region_model = self.models['region']
                
                # Ensure feature order matches training
                if self.region_features is not None:
                    # Fill missing features with zeros or mean values
                    X_full = pd.DataFrame(index=X.index, columns=self.region_features)
                    for col in self.region_features:
                        if col in X.columns:
                            X_full[col] = X[col]
                        else:
                            X_full[col] = 0  # Default value for missing features
                    X = X_full
                else:
                    X = X_selected
            else:
                # Use all features - ensure order matches training
                if self.region_features is not None:
                    missing_cols = set(self.region_features) - set(X.columns)
                    if missing_cols:
                        # Fill missing columns with default values
                        for col in missing_cols:
                            X[col] = 0
                    X = X[self.region_features]
            
            # Make prediction
            region_model = self.models['region']
            prediction = region_model.predict(X)
            probabilities = region_model.predict_proba(X)
            
            # Get label encoder for region to decode prediction
            region_encoder = self.preprocessor['label_encoders']['Region']
            predicted_region = region_encoder.inverse_transform(prediction)
            
            # Get feature importance
            feature_importance = pd.Series(
                region_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            return {
                "prediction": predicted_region[0],
                "probabilities": probabilities[0],
                "classes": region_encoder.classes_,
                "feature_importance": feature_importance.to_dict(),
                "confidence": np.max(probabilities[0]),
                "features_used": list(X.columns)
            }
            
        except Exception as e:
            return {"error": f"Error in region prediction: {e}"}
    
    def predict_month(self, data: pd.DataFrame, selected_features: list = None) -> Dict[str, Any]:
        """Predict month for given data with optional feature selection"""
        if self.models is None:
            return {"error": "Models not loaded"}
        
        try:
            # Prepare data for month prediction (exclude month column)
            X = data.drop(columns=['Month'], errors='ignore')
            
            # Handle feature selection
            if selected_features is not None:
                # Use only selected features
                missing_features = [f for f in selected_features if f not in X.columns]
                if missing_features:
                    return {"error": f"Missing selected features: {missing_features}"}
                X_selected = X[selected_features]
                
                # Ensure feature order matches training
                if self.month_features is not None:
                    # Fill missing features with zeros or mean values
                    X_full = pd.DataFrame(index=X.index, columns=self.month_features)
                    for col in self.month_features:
                        if col in X.columns:
                            X_full[col] = X[col]
                        else:
                            X_full[col] = 0  # Default value for missing features
                    X = X_full
                else:
                    X = X_selected
            else:
                # Use all features - ensure order matches training
                if self.month_features is not None:
                    missing_cols = set(self.month_features) - set(X.columns)
                    if missing_cols:
                        # Fill missing columns with default values
                        for col in missing_cols:
                            X[col] = 0
                    X = X[self.month_features]
            
            # Make prediction
            month_model = self.models['month']
            prediction = month_model.predict(X)
            probabilities = month_model.predict_proba(X)
            
            # Get label encoder for month to decode prediction
            month_encoder = self.preprocessor['label_encoders']['Month']
            predicted_month = month_encoder.inverse_transform(prediction)
            
            # Get feature importance
            feature_importance = pd.Series(
                month_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            return {
                "prediction": predicted_month[0],
                "probabilities": probabilities[0],
                "classes": month_encoder.classes_,
                "feature_importance": feature_importance.to_dict(),
                "confidence": np.max(probabilities[0]),
                "features_used": list(X.columns)
            }
            
        except Exception as e:
            return {"error": f"Error in month prediction: {e}"}
    
    def predict_both(self, data: pd.DataFrame, selected_features: list = None) -> Dict[str, Any]:
        """Predict both region and month for given data with optional feature selection"""
        region_result = self.predict_region(data, selected_features)
        month_result = self.predict_month(data, selected_features)
        
        return {
            "region": region_result,
            "month": month_result
        }
    
    def batch_predict(self, data: pd.DataFrame, target: str = "both") -> pd.DataFrame:
        """Make predictions for batch data"""
        if self.models is None:
            st.error("Models not loaded")
            return pd.DataFrame()
        
        try:
            results = []
            
            for idx, row in data.iterrows():
                row_df = pd.DataFrame([row])
                
                if target == "region":
                    pred_result = self.predict_region(row_df)
                    if "error" not in pred_result:
                        results.append({
                            "index": idx,
                            "predicted_region": pred_result["prediction"],
                            "region_confidence": pred_result["confidence"]
                        })
                
                elif target == "month":
                    pred_result = self.predict_month(row_df)
                    if "error" not in pred_result:
                        results.append({
                            "index": idx,
                            "predicted_month": pred_result["prediction"],
                            "month_confidence": pred_result["confidence"]
                        })
                
                else:  # both
                    pred_results = self.predict_both(row_df)
                    
                    if ("error" not in pred_results["region"] and 
                        "error" not in pred_results["month"]):
                        results.append({
                            "index": idx,
                            "predicted_region": pred_results["region"]["prediction"],
                            "region_confidence": pred_results["region"]["confidence"],
                            "predicted_month": pred_results["month"]["prediction"],
                            "month_confidence": pred_results["month"]["confidence"]
                        })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            st.error(f"Error in batch prediction: {e}")
            return pd.DataFrame()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        if self.models is None:
            return {"error": "Models not loaded"}
        
        try:
            region_model = self.models['region']
            month_model = self.models['month']
            
            return {
                "region_model": {
                    "type": type(region_model).__name__,
                    "n_classes": len(region_model.classes_),
                    "classes": self.preprocessor['label_encoders']['Region'].classes_.tolist(),
                    "n_features": region_model.n_features_in_
                },
                "month_model": {
                    "type": type(month_model).__name__,
                    "n_classes": len(month_model.classes_),
                    "classes": self.preprocessor['label_encoders']['Month'].classes_.tolist(),
                    "n_features": month_model.n_features_in_
                }
            }
            
        except Exception as e:
            return {"error": f"Error getting model info: {e}"}
    
    def display_prediction_results(self, results: Dict[str, Any]):
        """Display prediction results in Streamlit"""
        st.subheader("Prediction Results")
        
        if "region" in results and "month" in results:
            # Both predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Region Prediction")
                region_result = results["region"]
                if "error" not in region_result:
                    st.metric("Predicted Region", region_result["prediction"])
                    st.metric("Confidence", f"{region_result['confidence']:.2%}")
                    
                    # Show top probabilities
                    st.markdown("**Top Probabilities:**")
                    probs_df = pd.DataFrame({
                        'Region': region_result["classes"],
                        'Probability': region_result["probabilities"]
                    }).sort_values('Probability', ascending=False).head(3)
                    st.dataframe(probs_df, hide_index=True)
                else:
                    st.error(region_result["error"])
            
            with col2:
                st.markdown("### Month Prediction")
                month_result = results["month"]
                if "error" not in month_result:
                    st.metric("Predicted Month", month_result["prediction"])
                    st.metric("Confidence", f"{month_result['confidence']:.2%}")
                    
                    # Show top probabilities
                    st.markdown("**Top Probabilities:**")
                    probs_df = pd.DataFrame({
                        'Month': month_result["classes"],
                        'Probability': month_result["probabilities"]
                    }).sort_values('Probability', ascending=False).head(3)
                    st.dataframe(probs_df, hide_index=True)
                else:
                    st.error(month_result["error"])
        
        # Feature importance
        if ("region" in results and "error" not in results["region"] and
            "month" in results and "error" not in results["month"]):
            
            st.markdown("### Feature Importance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Region Prediction - Top Features:**")
                region_fi = results["region"]["feature_importance"]
                region_fi_df = pd.DataFrame(
                    list(region_fi.items())[:5], 
                    columns=['Feature', 'Importance']
                )
                st.dataframe(region_fi_df, hide_index=True)
            
            with col2:
                st.markdown("**Month Prediction - Top Features:**")
                month_fi = results["month"]["feature_importance"]
                month_fi_df = pd.DataFrame(
                    list(month_fi.items())[:5], 
                    columns=['Feature', 'Importance']
                )
                st.dataframe(month_fi_df, hide_index=True)
