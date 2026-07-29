# SMS Spam Classification
A full-stack machine learning web application for classifying SMS messages into Spam or Ham, visualizing dataset distributions, and running real-time live prediction tests using Python, Scikit-Learn, NLTK, and Streamlit.

## Project Structure
```
SpamClassifier/
├── data/
│   └── SMSSpamCollection      # SMS Spam dataset (5,574 messages)
├── scripts/
│   └── Doc2Spam.py            # Standalone analysis & model experimentation script
├── .streamlit/                # Streamlit UI theme & configuration
├── app.py                     # Main Streamlit web application & classification pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Features
### 1. Project Overview & Abstract
- Comprehensive abstract explaining text tokenization, lemmatization, and Naive Bayes classification
- Interactive visual metrics ribbon displaying raw counts, balancing ratios, and model type
- Pipeline workflow panel detailing each data processing and inference stage

### 2. Data Loading & Visualization
- Automated class imbalance resolution via oversampling of the minority Spam class
- Custom feature engineering (`word_count`, `contains_currency_symbols`, `contains_number`)
- Seaborn count plots and histogram distributions for visual exploratory data analysis

### 3. Model Training & Evaluation
- Scikit-Learn Multinomial Naive Bayes model training with train/test splitting (80/20)
- Real-time confusion matrix heatmap grid visualization
- Complete classification report metrics (precision, recall, F1-score)

### 4. Live Prediction Test
- Interactive text box for testing real-time SMS message classification
- Instant inference using TF-IDF feature weighting and lemmatization
- Immediate visual feedback alerts (Spam vs. Ham)

### 5. Full Code Explorer & Explanations
- Interactive Python code blocks that users can execute directly in the browser
- Tech-powered tooltips and hover highlights explaining ML terminology
- Step-by-step block explanations for educational breakdowns

## Technologies Used
- **Backend & ML:** Python 3, Scikit-Learn, NLTK (Natural Language Toolkit)
- **Data Manipulation & Visualization:** NumPy, Pandas, Matplotlib, Seaborn
- **Frontend UI & Web Server:** Streamlit (with Glassmorphism & mobile-responsive CSS)
- **Dependency Management:** pip / requirements.txt

## Dataset Setup
### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### Installation Steps
1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd SpamClassifier
   ```
2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset Configuration**
   - The dataset (`SMSSpamCollection`) is stored in tab-separated format inside the `data/` directory:
     ```python
     data_path = "data/SMSSpamCollection"
     ```
   - NLTK corpora (`stopwords` and `wordnet`) are automatically downloaded silently at runtime.

## Building the Project
### Using Pip
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # Windows

# Install project packages
pip install -r requirements.txt
```

## Running the Application
1. **Start the Streamlit Server**
   ```bash
   streamlit run app.py
   ```
2. **Access the application**
   - Open your web browser and navigate to:  
     `http://localhost:8501/`

## Default Test Samples
### Sample Spam Message
- **Text:** `"IMPORTANT - You can be entitled up to $3160 from sis-sold PPI on a credit card or loan, Please check."`
- **Expected Prediction:** `🚨 This is a SPAM message.`

### Sample Ham Message
- **Text:** `"Come to think of it, I have never got a spam message before."`
- **Expected Prediction:** `✅ This is a HAM (normal) message.`

## Application Pages & Modules
### Navigation
- `GET /Project Overview` - Display project abstract, core objectives, and workflow steps
- `GET /1. Data Loading & Visualization` - Data balancing and feature distribution charts
- `GET /2. Model Training & Evaluation` - Model fitting and confusion matrix evaluation
- `GET /3. Live Prediction Test` - Live interactive text input prediction
- `GET /4. Full Code Explorer` - Interactive Python execution blocks with explanations
- `GET /5. View Raw Source Code` - Direct inspection of source code

## Dataset Schema & Engineered Features
### Raw Dataset (`SMSSpamCollection`)
- `label` (`ham` -> 0, `spam` -> 1)
- `message` (Raw text SMS string)

### Engineered Features
- `word_count` - Total number of words in message
- `contains_currency_symbols` - Binary flag (1 if `$`, `€`, `₹`, `¥`, or `₺` present, else 0)
- `contains_number` - Binary flag (1 if digits `0-9` present, else 0)

### Model Features
- `TF-IDF Word Vectors` - Term Frequency-Inverse Document Frequency matrix (`max_features=500`)

## Classification Categories Supported
- **Ham:** Legitimate personal or transactional text messages
- **Spam:** Unsolicited promotional spam, scams, lottery winnings, or phishing links

## Security & Reliability Features
- Class imbalance correction using Oversampling
- TF-IDF feature weighting (`max_features=500`)
- Regex-based special character & numeric filtering
- Stopword removal via NLTK English stopwords
- Semantic root preservation via WordNet Lemmatization
- Mobile-responsive Glassmorphism UI layout

## Future Enhancements
- Support for multi-lingual spam detection
- Real-time SMS API integration for live mobile spam blocking
- Advanced transformers (BERT / RoBERTa) for contextual embedding classification
- Exporting model weights (`.pkl` / `joblib`) for microservice deployment
- Custom user feedback loop to report misclassified SMS messages
- Dark/Light mode toggle switch
- Advanced reporting and ROC-AUC statistics

## Troubleshooting
### NLTK LookupError / Missing Corpus
If NLTK data fails to download automatically, run:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Dataset File Not Found
Ensure `SMSSpamCollection` is located in the `data/` directory or root directory:
```bash
# Check directory contents
dir data
```

### Streamlit Port Conflicts
If port `8501` is already in use, start the app on an alternative port:
```bash
streamlit run app.py --server.port 8502
```

## Performance Optimization
- `@st.cache_resource` caching implemented for dataset loading and Naive Bayes training
- Vectorized string operations using Pandas and NumPy
- Limited `TfidfVectorizer` to 500 features to optimize memory and speed up live inference
- Lightweight CSS animations optimized for mobile and desktop rendering

## License
This project is open source and available under the MIT License.

## Author
Developed by: Noor Mohammad

## Support
For issues and questions, please refer to the documentation or contact support.

## Deployment Checklist
- [x] `SMSSpamCollection` dataset placed in `data/` directory
- [x] Dataset path fallback configured
- [x] `requirements.txt` dependencies resolved
- [x] NLTK corpora (`stopwords`, `wordnet`) download automated
- [x] Streamlit web server configured and tested
- [x] Glassmorphism UI & mobile responsiveness verified
- [x] Multinomial Naive Bayes model accuracy verified
- [x] Interactive Code Explorer blocks tested
- [x] Zero changes made to existing UI or functionality
