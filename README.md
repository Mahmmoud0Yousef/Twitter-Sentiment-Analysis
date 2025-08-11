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
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## 📊 Dataset

### Twitter Sentiment Analysis Dataset

This project uses the **Twitter Entity Sentiment Analysis Dataset** which contains entity-level sentiment analysis for tweets.

**Dataset Source**: [Twitter Entity Sentiment Analysis - Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

### Dataset Description

- **Size**: ~74,000 labeled tweets
- **Training Data**: ~70,000 messages (`twitter_training.csv`)
- **Validation Data**: ~1,000 messages (`twitter_validation.csv`)
- **Languages**: Multi-lingual tweets (primarily English)

### Data Format

| Column | Description |
|--------|-------------|
| `ID` | Unique identifier for each tweet |
| `Entity` | The entity/company being referenced |
| `Labels` | Sentiment label (4 categories) |
| `Text` | The actual tweet content |

### Sentiment Categories

- **Irrelevant** (0): Tweet doesn't express sentiment towards the entity
- **Negative** (1): Negative sentiment towards the entity
- **Neutral** (2): Neutral sentiment towards the entity  
- **Positive** (3): Positive sentiment towards the entity

### Data Download

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
2. Download `twitter_training.csv` and `twitter_validation.csv`
3. Place both files in your project root directory

**Note**: You'll need a Kaggle account to download the dataset. The dataset is free and publicly available.

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
pip install -r requirements.txt
```

### Required Dependencies

Create a `requirements.txt` file with the following dependencies:

```txt
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
nltk>=3.6.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
flask>=2.0.0
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

4. **Download and prepare your data**
   - Visit [Kaggle dataset page](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
   - Download `twitter_training.csv` and `twitter_validation.csv`
   - Place both files in your project root directory
   - Ensure data format: `ID`, `Entity`, `Labels`, `Text` columns


### Text Preprocessing Pipeline

The preprocessing includes:
- Converting to lowercase
- Removing HTML tags and URLs
- Removing punctuation and digits
- Filtering stopwords (preserving negation words)
- Tokenization and lemmatization
- Sequence padding and truncation

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
- **Epochs**: 100 (with early stopping, patience=11)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Class Weighting**: Balanced weights for handling imbalanced data

### Class Labels
- **0**: Irrelevant
- **1**: Negative  
- **2**: Neutral
- **3**: Positive

## 🛠️ Technical Details

### Custom Transformer Implementation

The model implements a complete Transformer encoder with:

- **Multi-Head Attention**: Captures different types of relationships in text
- **Positional Encoding**: Adds position information to embeddings using sin/cos functions
- **Layer Normalization**: Stabilizes training with epsilon=1e-6
- **Residual Connections**: Prevents vanishing gradients
- **Dropout**: Prevents overfitting (rate=0.1)
- **Feed-Forward Networks**: Two linear transformations with ReLU activation

### Text Processing Features

- **Negation Preservation**: Maintains important negation words during stopword removal
- **Balanced Class Weights**: Automatically calculates weights for imbalanced classes
- **Sequence Padding**: Handles variable-length inputs (max_len=25)
- **OOV Token Handling**: Manages out-of-vocabulary words with `<OOV>` token

## 🎨 Web Interface

The Flask application provides:
- Clean, responsive Bootstrap-based UI
- Real-time sentiment prediction
- Confidence scores for predictions
- Easy-to-use text input interface
- Error handling for empty inputs

## 📈 Visualization

The training script generates:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix heatmap with class labels
- Text length distribution histogram
- Classification report with precision, recall, and F1-score

## 🔧 Customization

### Modifying Model Architecture

To change the model configuration, edit these parameters in `main_training.py`:

```python
# Model hyperparameters
vocab_size = len(tokenizer.word_index) + 1
max_len = 25              # Maximum sequence length
num_layers = 6            # Number of transformer layers
embed_dim = 128           # Embedding dimension
num_heads = 8             # Number of attention heads
ff_dim = 256             # Feed-forward layer dimension
num_classes = 4          # Number of output classes
dropout_rate = 0.1       # Dropout rate
```

### Adding New Classes

1. Update the number of classes in the model initialization
2. Modify the label encoder mapping
3. Update the labels list in evaluation code
4. Adjust the final Dense layer output dimension

### Preprocessing Customization

Modify the `Preprocessing_text()` and `tokenize_lemmatize()` functions to:
- Add custom text cleaning rules
- Include additional stopwords or negation words
- Change tokenization strategy
- Adjust sequence length based on your data

## 🚀 Deployment

### Local Deployment
The Flask app runs locally by default. For production deployment:

1. Set `debug=False` in `app.py`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
3. Configure environment variables for model and tokenizer paths

### Model Export Options
The trained model is saved in TensorFlow format and can be:
- **TensorFlow Serving**: For scalable model serving
- **TensorFlow Lite**: For mobile and edge deployment
- **ONNX Format**: For cross-platform deployment
- **TensorFlow.js**: For browser-based inference

## 🧪 Testing

To test the model performance:

```python
# Load the model and test
model = tf.keras.models.load_model("model_sentiment")
test_loss, test_accuracy = model.evaluate(x_test_pad, y_test_categorical)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

