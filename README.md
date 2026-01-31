# Sentiment Analysis with LSTM
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit)

This project implements an end-to-end sentiment analysis system capable of classifying raw English text reviews as positive or negative. It leverages a custom-trained Bidirectional LSTM (Long Short-Term Memory) neural network trained on the IMDB movie reviews dataset.

Unlike standard tutorials that use pre-indexed datasets, this project reconstructs the dataset from raw text files, trains a custom tokenizer, and deploys the final model as an interactive local web application.

---

## Features

* **End-to-End Pipeline:** Handles raw text loading, cleaning, tokenization, padding, and inference.
* **Deep Learning Architecture:** Uses a Bidirectional LSTM network to capture context from both past and future words in a sentence.
* **Custom Tokenizer:** Builds a vocabulary from scratch on the training data, ensuring the model is not dependent on external pre-built indices.
* **Interactive Web App:** Deploys the model using Streamlit for real-time user testing and visualization.
* **Training Safeguards:** Implements EarlyStopping and ModelCheckpoint to prevent overfitting and save only the best-performing model weights.

---

## Project Files

* **`app.py`**
    The main application script powered by Streamlit. It loads the saved model and tokenizer to provide a user-friendly interface for real-time sentiment prediction.
    
* **`train_model.py`**
    The complete training script. It handles downloading the raw IMDB data, preprocessing the text, building the Bi-LSTM architecture, training the model, and saving the artifacts.

* **`best_model.h5`**
    The trained deep learning model file (Keras H5 format). This version contains the weights that achieved the highest validation accuracy during training.

* **`tokenizer.pickle`**
    The serialized Python object containing the word-to-index dictionary. This ensures that the web app processes text exactly the same way the model was trained.

* **`requirements.txt`**
    A list of necessary Python libraries required to run the project.

---

## Author

**Ishan Zadbuke**
Developed as a comprehensive learning exercise in applied Natural Language Processing (NLP) and Deep Learning system design.
