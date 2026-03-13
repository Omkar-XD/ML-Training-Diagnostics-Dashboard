# ML Training Diagnostics Dashboard

## Summary

A dashboard that trains machine learning models and analyzes training behavior through loss curves, validation trends, and diagnostic flags such as overfitting, underfitting, and unstable training.

## Tech Stack

Backend: FastAPI
ML libraries: scikit-learn
Frontend: React + Chart.js

## How to Run

### Backend

cd backend
pip install -r requirements.txt
uvicorn app:app --reload

### Frontend

cd frontend
npm install
npm run dev

## Data

Place raw dataset in:

/data/raw.csv

After preprocessing the cleaned dataset is saved as:

/data/cleaned.csv

## Features

* CSV upload
* Automatic EDA summary
* Missing value handling
* Train models:

  * Linear Regression
  * Decision Tree
  * MLP
* Training diagnostics:

  * Loss curves
  * Overfitting detection
  * Underfitting detection
  * Unstable training warnings

## Week-wise Concept Mapping

Week 1: Python modular code, functions, file handling
Week 2: Data cleaning, missing value handling
Week 3: Cost functions and gradient intuition
Week 4: Bias-variance tradeoff
Week 5: Hyperparameter experiments
Week 6: Loss curves and training diagnostics

## Deliverables

* training_analysis.md
* logs/ (JSON experiment files)
* dataset files
