# *Text messages spam classifier with TF-IDF and Naive Bayes*
A binary text classifier built with **scikit-learn** that classifies text messages as **spam** or **ham**. Nothing more than a **self-learning ML project** to improve my skills in **NLP**. In the notebook file you will find code for
- Cleaning and preprocessing raw text;
- Converting messages to numeric features using **TF-IDF vectorization**;
- Training a **Multinomial Naive Bayes** classifier;
- Evaluating model performance with metrics and confusion matrix.
You will also find some mathemathical background about the tools in question.

## Dataset
Source: [SMS Spam Collection](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
The dataset is really lightweight and made of two columns of tab-separated values. Labels are in the first column, every message is either marked as `spam` or `ham`; the second column is for SMS content.

## How to open and run the notebook locally
1. Clone and open the repo
```bash
   git clone https://github.com/your-username/spam-classifier.git
   cd spam-classifier
```
2. *Create a virtual environment (optional)*
```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
```
3. Install dependencies
```bash
   pip install -r requirements.txt
```
4. Open the notebook
```bash
   jupyter notebook spam_binary_classifier.ipynb
```

## How to run the pre-trained model locally
1. Clone and open the repo
```bash
   git clone https://github.com/your-username/spam-classifier.git
   cd spam-classifier
```
2. Install joblib (if not already installed)
```bash
   pip install joblib
```
3. Load the model and the vectorizer in a local Python script
```python
   import joblib
   vectorizer = joblib.load('res/tfidf_vectorizer.joblib')
   model = joblib.load('res/spam_classifier_model.joblib')
```
4. Example usage
```python
   msg = "Congratulations! You've won a free ticket to Bahamas!"
   msg = msg.lower()
   msg = re.sub(f'{re.escape(string.punctuation)}', '', msg)
   msg_vec = vectorizer.transform([msg])
   prediction = model.predict(msg_vec)
   print("Spam" if prediction[0] == 1 else "Ham")
```

## Model evaluation
The project has a **TF-IDF vectorizer** configured with both unigrams and bigrams (1-2 grams) to convert text messages into numerical features. It then applies a **Multinomial Naive Bayes algorithm** The model’s performance is evaluated using  **accuracy**, **precision**, **recall**, and **F1-score**. You can also find a confusion matrix to visualize the number of correct and incorrect predictions for both spam and ham classes.

## Thank you!
This project is part of my learning journey in ML and AI 🙂. Feel free to connect or suggest improvements.
[LinkedIn](https://www.linkedin.com/in/gabriele-lobello/), [e-mail](mailto:gabrielelobello@outlook.com)