# SPAM EMAIL DETECTION SYSTEM
## MCA Mini Project Report

### 1. Abstract
Spam emails and unwanted messages can waste time and may contain misleading offers or unsafe links. This project demonstrates a machine-learning approach to classifying text messages as spam or not spam. The system uses TF-IDF to represent text numerically and a Multinomial Naive Bayes classifier to learn from labelled examples. A Streamlit interface allows a user to enter a message and view the predicted category. The project is designed for academic learning and does not guarantee the safety or legitimacy of a message.

### 2. Introduction
Email is widely used for personal, educational, and business communication. Spam messages are unwanted messages often sent in bulk. Manual identification can be difficult because messages vary in wording. Machine learning can learn patterns from labelled examples and use them to classify new text.

### 3. Problem Statement
To develop a simple application that accepts email/message text and predicts whether it is spam or not spam using Natural Language Processing and machine learning.

### 4. Objectives
- Understand a basic text-classification workflow.
- Convert text into numeric features using TF-IDF.
- Train a Multinomial Naive Bayes classifier.
- Evaluate the model using a held-out test split.
- Provide a simple web interface for testing messages.

### 5. Scope
The application is intended as an educational prototype. It processes text entered by the user. It does not connect to a real email inbox or automatically block messages.

### 6. Technologies Used
- Python: main programming language
- Pandas: reading and preparing CSV data
- Scikit-learn: TF-IDF, model training and evaluation
- Streamlit: web interface
- Joblib: saving and loading trained model files
- VS Code: development environment

### 7. System Architecture
User enters message → Streamlit interface → TF-IDF vectorizer → Naive Bayes classifier → prediction and confidence displayed to user.

### 8. Methodology
1. Load labelled text data from `spam.csv`.
2. Clean missing and invalid rows.
3. Split data into training and testing subsets.
4. Fit TF-IDF on the training text and transform the test text.
5. Train Multinomial Naive Bayes.
6. Evaluate predictions on the held-out test set.
7. Save the trained model and vectorizer.
8. Load the saved files in the web application and classify user input.

### 9. Testing
Test the application with multiple examples, including promotional messages, ordinary personal messages, and ambiguous text. Record the model's actual output after running the program. Do not claim a fixed accuracy before training; the script prints the measured test accuracy and classification report.

### 10. Expected Result
The application should display either SPAM or NOT SPAM for non-empty input and show an estimated confidence. Results depend on the training data and the wording of the test message.

### 11. Limitations
The included dataset is small and only for demonstration. Real spam detection needs more varied and current data. The model may misclassify messages, and confidence should not be interpreted as proof that an email is safe.

### 12. Future Scope
- Use a larger public dataset.
- Compare different classification algorithms.
- Add a results dashboard with precision, recall, F1-score and confusion matrix.
- Support multiple languages.
- Integrate with a mail client only after careful privacy and security review.

### 13. Conclusion
This mini project demonstrates the core steps of a text-classification application: data preparation, text vectorization, model training, evaluation, and deployment through a small web interface. It provides a practical introduction to NLP and machine learning.
