# Spam Email Detection System — MCA Mini Project

## 1. Project overview
This project predicts whether an email/message looks like **spam** or **not spam**.
It uses Natural Language Processing (NLP), TF-IDF text representation, and a
Multinomial Naive Bayes machine-learning classifier. A simple web interface is made
with Streamlit.

## 2. Features
- Enter email/message text in a browser.
- Predict SPAM or NOT SPAM.
- Show a model confidence estimate.
- Train and evaluate the model with a labelled CSV dataset.
- Beginner-friendly Python code.

## 3. Software requirements
- Python 3.10 or newer recommended
- VS Code
- Internet connection for the first-time package installation

## 4. Project files
- `app.py` — web interface
- `train_model.py` — trains and evaluates the ML model
- `spam.csv` — small teaching/demo dataset
- `requirements.txt` — Python packages
- `spam_model.joblib` and `tfidf_vectorizer.joblib` — created after training

## 5. Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open VS Code.
3. Select **File → Open Folder** and choose `Spam_Email_Detection_MCA`.
4. Open **Terminal → New Terminal**.
5. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
6. Activate it in Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
7. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
8. Train the model:
   ```bash
   python train_model.py
   ```
9. Start the website:
   ```bash
   python -m streamlit run app.py
   ```
10. A browser tab should open. If it does not, copy the local URL printed in the terminal (usually `http://localhost:8501`) into your browser.

## 6. How to test
Try these example messages:
- `Congratulations! You have won a free prize. Click now to claim your reward!`
- `Hi, our meeting is scheduled for 10 AM tomorrow. Please bring the report.`
- `URGENT! Claim your cash bonus now. Limited time offer!`

The demo dataset is small and intended for learning. Predictions can be wrong. For a stronger academic project, use a larger, properly licensed public spam dataset and report results from a held-out test set.

## 7. How it works
1. The labelled CSV provides example messages and labels.
2. TF-IDF converts text into numerical features.
3. Naive Bayes learns patterns associated with spam and not-spam examples.
4. The trained model predicts the label of a new message.
5. Streamlit displays the prediction.

## 8. Limitations
- This is a classroom demonstration, not a production email filter.
- The small demo dataset is not representative of all real emails.
- Confidence is not a guarantee of correctness.
- Avoid entering private or confidential email content.

## 9. Suggested future improvements
- Train on a larger public dataset.
- Add precision, recall, F1-score and confusion matrix to a report.
- Compare Naive Bayes with Logistic Regression.
- Add multilingual text support and more realistic testing.
