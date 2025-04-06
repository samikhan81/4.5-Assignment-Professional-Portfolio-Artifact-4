# 4.5-Assignment-Professional-Portfolio-Artifact-4

**Sentiment Analysis of Social Media Posts Using NLP and LSTM**

A natural language processing (NLP) project that classifies social media posts (tweets) into **positive**, **negative**, or **neutral** sentiments using an LSTM neural network. Includes a Flask web interface for real-time predictions.

---

## 📌 Features
- **Text Preprocessing**: Cleans tweets by removing URLs, mentions, and hashtags, and converts emojis to text.
- **LSTM Model**: Deep learning model with 85% accuracy for sentiment classification.
- **Web Interface**: User-friendly Flask app for live predictions.
- **Bias Mitigation**: Handles class imbalance and sarcasm detection.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Twitter API keys (optional, for scraping new data)

### Steps
1. Clone the repository:
   ```bash
https://github.com/samikhan81/4.5-Assignment-Professional-Portfolio-Artifact-4/edit/main/README.md
   cd sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
   ```

4. Run the Flask app:
   ```bash
   python sentiment_analysis.py
   ```


---

## 📂 Dataset
- **Source**: Tweets scraped using the Twitter API (or manually labeled data).
- **Sample Data**:  

  | text                                  | sentiment |
  |---------------------------------------|-----------|
  | "Loving the new update! 😊"           | Positive  |
  | "This service is terrible. 😤"        | Negative  |
  | "Just booked my flight to NYC. 🛫"     | Neutral   |

---

## 🧠 Model Architecture

```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## 📊 Results

| Metric      | Positive | Negative | Neutral |
|-------------|----------|----------|---------|
| **Precision** | 0.87     | 0.83     | 0.79    |
| **Recall**    | 0.82     | 0.85     | 0.76    |


