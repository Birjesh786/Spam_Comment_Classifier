<p align="center"> <img width="1500px" src="https://s12.gifyu.com/images/SWBtr.gif"/> </p>

# Spam Comment Classifier 

Hello ML buddies, I've shared with you the Spam Classifier Model.

## What Part

What is Spam detection?


Spam detection is the process of locating and removing undesired or irrelevant content, generally from online comments, text messaging, or other digital communication channels like email. Spam detection aims to distinguish between messages that are valuable and legitimate and messages that are considered spam or garbage.


## Why Part

Why spam detection is most important?

Because it protects digital communication channels like email and messaging from undesired and potentially destructive content, spam detection is crucially vital. Inboxes can get overrun with spam messages, making it harder for users to locate legitimate and crucial information. Furthermore, spam frequently includes scams, phishing attempts, or dangerous links that jeopardize user security and privacy. Spam detection systems improve user experience, defend against cybersecurity risks, and aid in maintaining the integrity of online communication by filtering out spam. This ensures that legitimate messages reach their intended recipients while reducing the risks associated with unwanted and misleading content.

## How Part

### How to make this project?

In this project, I've used two approaches:

#### With labeled dataset: 

The spam detection effort is greatly simplified by having access to a labelled dataset. You can use supervised machine learning methods using a labelled dataset. These methods entail building a model from the dataset, where each case is classified as spam or not. The model picks up on the patterns, traits, and qualities that set spam apart from real messages. Once the model has been trained, this method makes spam detection accurate and effective. Additionally, by retraining the model with fresh data and fine-tuning its algorithms in accordance with changing spam patterns, you can continuously raise the model's accuracy.

#### Without labeled dataset:

When there isn't a labelled dataset available, spam identification is harder to do. It is still possible to use efficient spam filters, though. Using rule-based systems, which entail developing a set of predefined rules and heuristics to recognize spammy qualities in communications, is one strategy. These guidelines may include well-known spam patterns, dubious sender domains, or popular spam terms. This method can assist in removing some evident spam even if it might not be as accurate as machine learning using labelled data.













## Prediction functions for both approaches

Here I've shown how to create a function for predicting spam massenging.

### Approaches:

1. Given a labelled dataset

2. Sample or dummy dataset

#### Given labeled dataset:

See pratical example here 👉 [Given Dataset based Spam CLassification](https://github.com/Birjesh786/Spam_Comment_Classifier/blob/main/Training%20files/Given%20Dataset%20based%20Spam%20Classification%20Model.ipynb)

```bash
def analyze_input(input_text):
    # Preprocess the input text
    input_text = preprocess_text(input_text)
 
    # Transform the input using the trained TF-IDF  vectorizer
    input_vector = tfidf_vectorizer.transform(  [input_text])

    # Predict whether the input is spam or not
    prediction = clf.predict(input_vector)[0]

    # Count the total words in the input
    total_words = len(nltk.word_tokenize(input_text))

    # Count the number of spam words in the input      (words that contributed to spam classification)
    spam_words_count = sum(1 for word in nltk.word_tokenize(input_text) if word in top_spam_words)

    # Calculate the percentage of spam words in the input
    percentage_spam = (spam_words_count / total_words) * 100

    # Provide an explanation for the classification
    if prediction == 0:
        explanation = "Not spam"
        color = "\033[92m" # Green color
        emoji = "" # Happy emoji
    else:
        explanation = "Spam"
        color = "\033[91m" # Red color
        emoji = "" # Sad emoji

    # Reset text formatting
    reset_format = "\033[0m"

    # Prepare the formatted explanation with emoji
    formatted_explanation = 
    f"{color}\033[3m  {explanation} {emoji  {reset_format}"

    # Prepare the result dictionary
    result = {
    "input_text": input_text,
    "total_words": total_words,
    "spam_words_count": spam_words_count,
    "percentage_spam": percentage_spam,
    "classification": formatted_explanation,
    "color": color
    }

return result
```

### Without labeled dataset:

#### 1. Classify each sentence in the paragraph: 

See pratical example here 👉 [Single Sentence_based Classification](https://github.com/Birjesh786/Spam_Comment_Classifier/blob/main/Training%20files/Single%20Sentence_based%20Classification%20Model.ipynb)


```bash

for sentence in sentences:
    # Preprocess the sentence
    sentence_vectorized = vectorizer.transform([sentence])

    # Predict whether the sentence is spam or not
    prediction = clf.predict(sentence_vectorized)

    # Count words in the sentence
    words = nltk.word_tokenize(sentence)
    word_count = len(words)

    # Identify spam words in the sentence
    spam_words_in_sentence = [word.lower() for word in  words if word.lower() in spam_words]
    spam_word_count = len(spam_words_in_sentence)

    # Define a prediction result
    prediction_result = "spam" if prediction[0] == 1    else "not spam"

    # Store information about the sentence
    sentence_info.append({
        'sentence': sentence,
        'word_count': word_count,
        'spam_word_count': spam_word_count,
        'prediction_result': prediction_result,
        'spam_words': spam_words_in_sentence,
})
```

#### 2. Create a function to classify a single sentence:

See pratical example here 👉 [Paragraph_based Classification](https://github.com/Birjesh786/Spam_Comment_Classifier/blob/main/Training%20files/Paragraph_based%20Classification%20Model.ipynb)


```bash
def classify_sentence(sentence):
    # Preprocess the sentence
    sentence_vectorized = vectorizer.transform( [sentence])

    # Predict whether the sentence is spam or not
    prediction = clf.predict(sentence_vectorized)
    
    # Count words in the sentence
    words = nltk.word_tokenize(sentence)
    word_count = len(words)

    # Identify spam words in the sentence
    spam_words_in_sentence = [word.lower() for word in  words if word.lower() in spam_words]
    spam_word_count = len(spam_words_in_sentence)

    # Define a prediction result based on the model's   prediction
    if prediction[0] == 1:
        prediction_result = "spam"
    else:
        prediction_result = "not spam"

    # Create a result dictionary
    result = {
        'sentence': sentence,
        'word_count': word_count,
        'spam_word_count': spam_word_count,
        'prediction_result': prediction_result,
}

if prediction_result == 'spam':
    result['spam_words'] = spam_words_in_sentence
return result
```


## Tech Stack (Language, Algorithms)

**Project Catagory:** ML + NLP

**Languages:** Python

**Platform:** Jupyter Notebook

**Algritham's:** NLTK, Logistic regression, Naive baise, Random forest.

## Support

If you like this project, please give us your opinion, and feel free to point out any potential areas for improvement. 

You can contact me at 📩 hire.brijesh@gmail.com, and I would definitely appreciate it.
  


