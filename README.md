# 🗨️ Sentiment Analysis using NLP

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![NLP](https://img.shields.io/badge/NLP-Natural_Language_Processing-green?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

A Natural Language Processing project that analyzes the sentiment of text data (positive, negative, or neutral) using machine learning techniques. Built with NLTK, TF-IDF vectorization, and classification algorithms for accurate sentiment detection.

## ✨ Features

- 📝 **Text Preprocessing**: Tokenization, stopword removal, lemmatization
- 🔢 **TF-IDF Vectorization**: Convert text to numerical features
- 🤖 **Multiple Classifiers**: Naive Bayes, Logistic Regression, SVM
- 📊 **Performance Metrics**: Accuracy, precision, recall, F1-score
- 📑 **Visualization**: Confusion matrix, word clouds, sentiment distribution
- ⚙️ **Custom Training**: Train on your own dataset
- 📥 **Export Model**: Save trained models for deployment

## 🛠️ Tech Stack

- **NLP Libraries**: NLTK, spaCy
- **ML Libraries**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Yuvraj6223/Sentimental-Analysis-using-NLP.git
cd Sentimental-Analysis-using-NLP
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 💻 Usage

### Basic Sentiment Analysis:

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze single text
text = "This product is amazing! I absolutely love it."
result = analyzer.predict(text)
print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")
# Output: Sentiment: Positive (Confidence: 92.5%)
```

### Training Custom Model:

```python
import pandas as pd
from sentiment_analyzer import SentimentAnalyzer

# Load your data
data = pd.read_csv('reviews.csv')  # Columns: 'text', 'sentiment'

# Train model
analyzer = SentimentAnalyzer()
analyzer.train(data['text'], data['sentiment'])

# Save model
analyzer.save_model('my_sentiment_model.pkl')
```

### Batch Processing:

```python
texts = [
    "I hate waiting in long queues!",
    "The service was okay, nothing special.",
    "Best experience ever! Highly recommend!"
]

results = analyzer.predict_batch(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]}... -> {result['sentiment']}")
```

## 📖 Preprocessing Pipeline

```
┌───────────────┐
│ Raw Text      │
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ Lowercase    │  Convert to lowercase
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ Remove       │  Remove punctuation, numbers
│ Special Chars│
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ Tokenization │  Split into words
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ Remove       │  Filter stopwords
│ Stopwords    │
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ Lemmatization│  Get base forms
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ TF-IDF       │  Vectorize
│ Vectorization│
└──────┬─────────┘
       │
       ▼
┌───────────────┐
│ Classifier   │  Predict sentiment
└───────────────┘
```

## 📊 Model Performance

### Classification Results:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | 85% | 84% | 86% | 85% |
| **Logistic Regression** | 88% | 87% | 89% | 88% |
| **SVM** | 87% | 86% | 88% | 87% |

## 📁 Project Structure

```
Sentimental-Analysis-using-NLP/
│
├── data/
│   ├── training_data.csv
│   └── test_data.csv
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── sentiment_analyzer.py   # Main analyzer class
├── preprocess.py           # Text preprocessing
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

## 📖 Supported Datasets

- IMDb Movie Reviews
- Twitter Sentiment140
- Amazon Product Reviews
- Custom CSV files (format: text, sentiment)

## ⚡ Future Enhancements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Aspect-based sentiment analysis
- [ ] Emotion detection (joy, anger, sadness)
- [ ] Multi-language support
- [ ] Real-time sentiment tracking
- [ ] API deployment with FastAPI

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/Enhancement`)
3. Commit changes (`git commit -m 'Add new model'`)
4. Push to branch (`git push origin feature/Enhancement`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NLTK](https://www.nltk.org/) for NLP tools
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms

## 📧 Contact

**Yuvraj V A** - [yuvrajva09@gmail.com](mailto:yuvrajva09@gmail.com)

Project Link: [https://github.com/Yuvraj6223/Sentimental-Analysis-using-NLP](https://github.com/Yuvraj6223/Sentimental-Analysis-using-NLP)

---

⭐ If you find this project helpful, please consider giving it a star!