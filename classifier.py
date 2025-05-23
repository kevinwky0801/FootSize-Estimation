"""
Enhanced image classifier for foot measurement system.
Uses SVM with improved preprocessing and feature extraction for better accuracy.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load
import logging
import time
from skimage.feature import hog
from skimage import exposure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (128, 128)  # Standard size for all images
CLASS_LABELS = {
    'normal_img': 0,
    'white_background_img': 1,
    'skin_background_img': 2,
    'pattern_background_img': 3,
    'curve_background_img': 4
}
MODEL_PATH = 'models/foot_classifier.joblib'
SCALER_PATH = 'models/scaler.joblib'
PCA_PATH = 'models/pca.joblib'

class ImageClassifier:
    def __init__(self, retrain: bool = False, use_pca: bool = True, use_hog: bool = True):
        """Initialize the classifier, loading or training model as needed.
        
        Args:
            retrain: Force retraining even if model exists
            use_pca: Use PCA for dimensionality reduction
            use_hog: Use HOG features for better texture representation
        """
        self.model = None
        self.scaler = None
        self.pca = None
        self.is_trained = False
        self.use_pca = use_pca
        self.use_hog = use_hog
        self.feature_shape = None
        
        if not retrain and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.load_model()
        else:
            self.train_model()
    
    def load_model(self) -> None:
        """Load pre-trained model, scaler, and PCA from disk."""
        try:
            self.model = load(MODEL_PATH)
            self.scaler = load(SCALER_PATH)
            if self.use_pca and os.path.exists(PCA_PATH):
                self.pca = load(PCA_PATH)
            self.is_trained = True
            logger.info("Loaded pre-trained model and scaler")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            self.train_model()
    
    def save_model(self) -> None:
        """Save trained model, scaler, and PCA to disk."""
        os.makedirs('models', exist_ok=True)
        try:
            dump(self.model, MODEL_PATH)
            dump(self.scaler, SCALER_PATH)
            if self.use_pca and self.pca is not None:
                dump(self.pca, PCA_PATH)
            logger.info("Saved model and components to disk")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_and_preprocess_data(self, data_dir: str = 'image_classified') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess training data from directory structure.
        
        Args:
            data_dir: Root directory containing categorized images
            
        Returns:
            Tuple of (features, labels) numpy arrays
        """
        X = []
        y = []
        
        for class_name, label in CLASS_LABELS.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Failed to read image: {img_path}")
                        
                    # Preprocessing pipeline
                    img = self.preprocess_image(img)
                    features = self.extract_features(img)
                    X.append(features)
                    y.append(label)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
        
        if not X:
            raise ValueError("No valid training images found")
            
        return np.array(X), np.array(y)
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Standard preprocessing for both training and prediction images.
        
        Args:
            img: Input BGR image
            
        Returns:
            Processed image
        """
        # Resize and normalize
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Convert to LAB color space which can be better for some tasks
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        
        # Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract features from preprocessed image.
        
        Args:
            img: Preprocessed image
            
        Returns:
            Feature vector
        """
        features = []
        
        # Color histogram features (using LAB space)
        hist_l = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
        hist_a = cv2.calcHist([img], [1], None, [16], [0, 256]).flatten()
        hist_b = cv2.calcHist([img], [2], None, [16], [0, 256]).flatten()
        features.extend(hist_l)
        features.extend(hist_a)
        features.extend(hist_b)
        
        if self.use_hog:
            # HOG features for texture
            gray = cv2.cvtColor(img, cv2.COLOR_LAB2GRAY)
            fd, hog_image = hog(
                gray,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                visualize=True,
                feature_vector=True
            )
            features.extend(fd)
            
            # Additional texture features
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            features.extend([np.mean(sobelx), np.std(sobelx), np.mean(sobely), np.std(sobely)])
        
        # Edge features
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), 100, 200)
        features.extend([np.mean(edges), np.std(edges)])
        
        return np.array(features)
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Train the SVM classifier with optional test split for evaluation.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        logger.info("Starting model training...")
        start_time = time.time()
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        self.feature_shape = X[0].shape
        
        # Split data if needed
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Dimensionality reduction
        if self.use_pca:
            self.pca = PCA(n_components=0.95, random_state=random_state)  # Keep 95% variance
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            logger.info(f"Reduced dimensions from {X_train.shape[1]} to {self.pca.n_components_}")
        
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }
        
        # Train SVM classifier with grid search
        grid_search = GridSearchCV(
            svm.SVC(probability=True, random_state=random_state),
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=3
        )
        
        logger.info("Starting grid search for best parameters...")
        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        self.is_trained = True
        
        # Evaluate if test split was created
        if test_size > 0:
            X_test_scaled = self.scaler.transform(X_test)
            if self.use_pca:
                X_test_scaled = self.pca.transform(X_test_scaled)
            
            y_pred = self.model.predict(X_test_scaled)
            
            logger.info("\nClassification Report:")
            logger.info(metrics.classification_report(y_test, y_pred, target_names=CLASS_LABELS.keys()))
            logger.info("\nConfusion Matrix:")
            logger.info(metrics.confusion_matrix(y_test, y_pred))
            
            # Log additional metrics
            logger.info(f"\nAccuracy: {metrics.accuracy_score(y_test, y_pred):.3f}")
            logger.info(f"Balanced Accuracy: {metrics.balanced_accuracy_score(y_test, y_pred):.3f}")
            
        self.save_model()
        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    def predict(self, image: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """
        Classify an input image.
        
        Args:
            image: Input BGR image array
            
        Returns:
            Tuple of (predicted_class, class_probabilities)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained")
            
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid")
        
        # Preprocess the image and extract features
        processed_img = self.preprocess_image(image)
        features = self.extract_features(processed_img)
        
        # Ensure feature shape matches training data
        if features.shape != self.feature_shape:
            raise ValueError(f"Feature shape mismatch. Expected {self.feature_shape}, got {features.shape}")
        
        # Scale features
        scaled_features = self.scaler.transform([features])
        
        # Apply PCA if used
        if self.use_pca and self.pca is not None:
            scaled_features = self.pca.transform(scaled_features)
        
        # Make prediction
        class_idx = self.model.predict(scaled_features)[0]
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        # Map probabilities to class names
        class_probs = {
            class_name: probabilities[idx]
            for class_name, idx in CLASS_LABELS.items()
        }
        
        return class_idx, class_probs

# Example usage
if __name__ == "__main__":
    # Initialize classifier (will load or train automatically)
    classifier = ImageClassifier(retrain=False, use_pca=True, use_hog=True)
    
    # Example prediction (replace with actual image)
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)  # Dummy image
    try:
        class_idx, class_probs = classifier.predict(test_image)
        class_name = [k for k, v in CLASS_LABELS.items() if v == class_idx][0]
        print(f"Predicted class: {class_name} with probabilities: {class_probs}")
    except Exception as e:
        print(f"Prediction failed: {e}")