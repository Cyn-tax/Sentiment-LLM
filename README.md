# Sentiment Analysis Dashboard

A comprehensive sentiment analysis tool that processes customer feedback and classifies sentiment using advanced language models.

## Features

- Real-time sentiment analysis for individual feedback texts
- Batch processing of CSV files with progress tracking
- Support for multiple AI models (OpenAI GPT-5 and Groq Llama-3.1-8b-instant)
- Detailed confusion matrix and performance metrics
- Interactive Streamlit dashboard with visual analytics

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Running the Application

**Start the Streamlit dashboard:**
```bash
python -m streamlit run Backend-Only/UI/Streamlit.py
```

**Run the OpenAI model directly (command line):**
```bash
python Backend-Only/Models/OpenAI.py
```

**Run the Groq model directly (command line):**
```bash
python Backend-Only/Models/Groq-llama.py
```

The Streamlit application will open in your default web browser at `http://localhost:8501`

## Usage

1. Select your preferred model (OpenAI or Groq) from the sidebar
2. Upload a CSV file with a `feedback_text` column
3. Process the file and view results with detailed metrics
4. Download results as JSON for further analysis

## Model Performance Recommendation

**To: Product Manager**

**Subject: Model Selection Recommendation for Sentiment Analysis**

After comprehensive evaluation of both available models, I recommend the following considerations for production deployment:

**OpenAI GPT-5 Performance:**
- Achieved an F1-score of approximately 96%
- Demonstrates superior accuracy in sentiment classification
- Significantly slower processing time per row
- Recommended for: High-stakes applications requiring maximum accuracy, low-volume processing, or scenarios where precision is critical

**Groq Llama-3.1-8b-instant Performance:**
- Achieved an F1-score of approximately 90%
- Faster processing time, trading accuracy for speed
- Suitable for high-volume, real-time applications
- Recommended for: High-throughput scenarios, cost-sensitive deployments, or applications where near-real-time responses are prioritized over marginal accuracy gains

**Recommendation:**
For production environments, consider a hybrid approach: utilize GPT-5 for critical sentiment analysis tasks where accuracy is paramount, and deploy Llama-3.1-8b-instant for high-volume, time-sensitive operations where a 6% accuracy differential is acceptable. The choice should align with business priorities, processing volume requirements, and cost constraints.

## Project Structure

```
Telus/
├── Backend-Only/
│   ├── Models/
│   │   ├── OpenAI.py          # OpenAI GPT-5 sentiment analyzer
│   │   └── Groq-llama.py      # Groq Llama-3.1-8b-instant analyzer
│   └── UI/
│       └── Streamlit.py       # Streamlit dashboard
├── Data.csv                   # Sample dataset
├── results.json               # Output results
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

