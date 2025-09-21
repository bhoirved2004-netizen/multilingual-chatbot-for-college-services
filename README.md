
# Multilingual College Chatbot (TF-IDF + Classifier)

This project contains a Streamlit app, a 500-row multilingual dataset (English, Hindi, Marathi),
and an improved dashboard for training and testing a TF-IDF + classifier-based chatbot for college services
(Admissions, Exams, Timetable, Results, Fees).

## Files
- app.py : Streamlit application
- college_chatbot_dataset_500.csv : Dataset with 500 Q&A rows
- requirements.txt : Python dependencies

## Run locally
1. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

## Notes
- The Hindi and Marathi translations are generated programmatically and may need human review for naturalness.
- For production use, consider using transformer-based models (IndicBERT/XLM-R) or embeddings + vector database.
