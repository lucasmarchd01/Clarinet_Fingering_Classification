# Hand Landmark Detection and Fingering Classification

## Overview

This repository contains Python scripts for hand landmark detection using MediaPipe and fingering classification using machine learning techniques. The hand landmark detection is performed in real-time using the MediaPipe library, while the fingering classification is based on pre-trained KNN and SVM classifiers.

## Contents

- `data_processing.py`: This script processes the CSV files containing hand landmark data collected from videos. It combines the data from multiple CSV files, generates unique frame numbers across videos, flattens the data, and reorders the columns for further processing.
- `classifier.py`: This script contains code for training KNN and SVM classifiers on the preprocessed hand landmark data. It loads the data, normalizes features, splits the data into training and testing sets, trains the classifiers, evaluates their performance, and saves the trained models and scaler.
- `note_overlay.py`: This script uses the MediaPipe library to perform real-time hand landmark detection. It draws landmarks on the detected hands and predicts fingerings using the pre-trained classifiers.


## Prerequisites

To start using on local machine:

- Clone this repository
- Create a conda environment `conda create -n "myenv" python=3.10` (or use your preffered virtual environment)
- run `conda install pip`
- run `pip install -r requirements.txt`

## Usage    

- Run the ython script for collecting hand landmark data.

`data_collection_overlay.py`

- Run the data_processing.py script to preprocess the hand landmark data:

`python data_processing.py`

- Run the classifier.py script to train the KNN and SVM classifiers:


`python classifier.py`

- Finally, run the note_overlay.py script to perform real-time hand landmark detection and fingering prediction:

`python note_overlay.py`

## File Structure

- `landmark_data_stable/`: Folder containing CSV files with hand landmark data collected from videos.
- `combined_data_filtered.csv`: Combined and filtered hand landmark data.
- `combined_data_unique_frames.csv`: Combined data with unique frame numbers across videos.
- `flattened_data.csv`: Flattened hand landmark data with reordered columns.
- `data_final.csv`: Final preprocessed data for training the classifiers.
