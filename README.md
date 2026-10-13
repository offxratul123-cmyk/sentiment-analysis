# Sentiment Analysis of Movie Reviews

A web-based sentiment analysis tool that classifies movie reviews as **Positive** or **Negative** using Machine Learning and Natural Language Processing.

## Overview

This project uses TF-IDF vectorization and Logistic Regression to classify IMDb movie reviews. It achieves **90.16% accuracy** on 50,000 IMDb reviews. A Streamlit web interface allows users to enter any English review and get instant sentiment prediction with confidence scores.

## Features

- Classifies English text as Positive or Negative
- Trained on 50,000 IMDb movie reviews
- 90.16% test accuracy
- Real-time prediction with confidence score
- Simple web interface built with Streamlit
- Handles HTML tags, punctuation, and stopwords
- Negation words (not, no, never) preserved for better accuracy

## Tech Stack

- Python 3.10
- scikit-learn (TF-IDF, Logistic Regression)
- NLTK (stopwords)
- pandas (data processing)
- Streamlit (web interface)
- joblib (model persistence)

## Project Structure

## Dataset

- **Name:** IMDB Dataset of 50K Movie Reviews
- **Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Size:** 50,000 reviews (25,000 positive, 25,000 negative)
- **Format:** CSV with two columns: `review`, `sentiment`

## How It Works

1. **Data Loading** — Load 50,000 IMDb reviews from CSV
2. **Text Preprocessing** — Remove HTML tags, punctuation, numbers, stopwords (except negations)
3. **Vectorization** — Convert text into numerical features using TF-IDF (20,000 features, bigrams)
4. **Model Training** — Train Logistic Regression on 40,000 reviews
5. **Evaluation** — Test on 10,000 unseen reviews
6. **Prediction** — Web interface classifies new reviews in real time

## Installation

### 1. Clone or download this project

```bash
git clone <your-repo-url>
cd sentiment-analysis