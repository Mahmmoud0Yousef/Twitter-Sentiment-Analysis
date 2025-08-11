# Sentiment Analysis Using Transformer Architecture

A sophisticated sentiment analysis system that uses a custom Transformer model to classify text into four sentiment categories: Irrelevant, Negative, Neutral, and Positive. The project includes both training pipeline and a web interface for real-time sentiment prediction.

## 🌟 Features

- **Custom Transformer Architecture**: Built from scratch using TensorFlow/Keras
- **Multi-Class Classification**: Classifies text into 4 sentiment categories
- **Advanced Text Preprocessing**: Comprehensive text cleaning and normalization
- **Class Balancing**: Handles imbalanced datasets using class weights
- **Web Interface**: Flask-based web application for easy interaction
- **Model Persistence**: Save and load trained models for deployment
- **Visualization**: Training metrics and confusion matrix visualization

## 🏗️ Architecture

### Transformer Model Components

1. **TransformerEncoderLayer**: Single encoder layer with multi-head attention
2. **TransformerEncoder**: Stack of multiple encoder layers  
3. **TransformerClassifier**: Complete model with embedding, positional encoding, and classification head

### Model Specifications
- **Vocabulary Size**: 20,000 words
- **Sequence Length**: 25 tokens
- **Embedding Dimension**: 128
- **Number of Layers**: 6
- **Attention Heads**: 8
- **Feed-Forward Dimension**: 256
- **Number of Classes**: 4 (Irrelevant, Negative, Neutral, Positive)

## 📁 Project Structure

```
sentiment-analysis/
│
├── app.py                   # Flask web application
├── model_classes.py         # Custom Transformer model classes
├── templates/
│   └── index.html          # Web interface template
├── data/
│   ├── twitter_training.csv    # Training dataset
│   └── twitter_validation.csv  # Validation dataset
├── models/
│   └── model_sentiment/    # Saved model directory
├── tokenizer.pkl           # Trained tokenizer
├── label_encoder.pkl       # Label encoder
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
pip install -r requirements.txt
```

### Required Dependencies

```python
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
tensorflow>=2.8.0
flask
pickle
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Prepare your data**
   - Place `twitter_training.csv` and `twitter_validation.csv` in the project directory
   - Ensure data format: `ID`, `Entity`, `Labels`, `Text` columns

## 🎯 Usage

### Training the Model

Run the training script to train your custom Transformer model:

```bash
python main_training.py
```

The script will:
- Load and preprocess the data
- Train the Transformer model
- Save the trained model and preprocessors
- Generate training visualizations
- Evaluate model performance

### Text Preprocessing Pipeline

The preprocessing includes:
- Converting to lowercase
- Removing HTML tags and URLs
- Removing punctuation and digits
- Filtering stopwords (preserving negation words)
- Tokenization and lemmatization

### Running the Web Application

Start the Flask web server:

```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface where you can:
- Enter text for sentiment analysis
- Get real-time predictions with confidence scores
- View results in a user-friendly interface

## 📊 Model Performance

### Training Configuration
- **Batch Size**: 128
- **Learning Rate**: 2e-4
- **Epochs**: 100 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

### Class Labels
- **0**: Irrelevant
- **1**: Negative  
- **2**: Neutral
- **3**: Positive

## 🛠️ Technical Details

### Custom Transformer Implementation

The model implements a complete Transformer encoder with:

- **Multi-Head Attention**: Captures different types of relationships in text
- **Positional Encoding**: Adds position information to embeddings
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Prevents vanishing gradients
- **Dropout**: Prevents overfitting

### Text Processing Features

- **Negation Preservation**: Maintains important negation words during stopword removal
- **Balanced Class Weights**: Automatically calculates weights for imbalanced classes
- **Sequence Padding**: Handles variable-length inputs
- **OOV Token Handling**: Manages out-of-vocabulary words

## 🎨 Web Interface

The Flask application provides:
- Clean, responsive Bootstrap-based UI
- Real-time sentiment prediction
- Confidence scores for predictions
- Easy-to-use text input interface

## 📈 Visualization

The training script generates:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix heatmap
- Text length distribution histogram

## 🔧 Customization

```

## 🚀 Deployment

### Local Deployment
The Flask app runs locally by default. For production deployment:

1. Set `debug=False` in `app.py`
2. Use a production WSGI server like Gunicorn
3. Configure environment variables for paths

### Model Export
The trained model is saved in TensorFlow format and can be:
- Loaded for inference
- Converted to TensorFlow Lite for mobile deployment
- Exported to ONNX format for cross-platform deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.



**Note**: Make sure to have sufficient computational resources for training. The model uses attention mechanisms which can be memory-intensive for longer sequences.
