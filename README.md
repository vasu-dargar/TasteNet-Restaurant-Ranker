# TasteNet — Restaurant Ranker
A deep learning and NLP-powered framework that ranks restaurants intelligently by combining numeric ratings and sentiment-analyzed text reviews. Instead of relying solely on rating averages, TasteNet predicts sentiment-based ratings, handles linguistic nuances, and computes a confidence-based ranking score for fair and reliable comparisons.

# Project Motivation
Choosing a good restaurant—especially while traveling or for special events—can be frustrating when ratings alone are misleading. Reviews may be emotional, sarcastic, biased, or inconsistent. TasteNet solves this by interpreting subjective reviews accurately and converting them into meaningful ranking signals.

# Key Features
    1-Semantic preprocessing to reduce sarcasm, slang, and ambiguous language
    2-Attention-based BiGRU sentiment model trained on reviews with numeric ratings
    3-Predicts ratings for text-only reviews
    4-Combines real, predicted, and independent ratings
    5-Confidence-based ranking metric using:
      a) Average rating
      b) Standard deviation (rating spread)
      c) Number of reviews
    6-A generalized summary is provided for each restaurant based on semantic analysis of each restaurant's reviews.
    
    Produces more consistent and trustworthy restaurant rankings compared to raw averages.

# System Architecture

                    Raw Reviews & Ratings
                            ↓
    Semantic Processing (Clean, Remove Slang/Sarcasm, Normalize)
                            ↓
            BiGRU + Attention Sentiment Model
                            ↓
      Predict Numeric Ratings for Text-only Reviews
                            ↓
                    Merge All Ratings
                            ↓
                Confidence Score Computation
                            ↓
                  Final Restaurant Ranking

# How to Run    
    # Install dependencies
        ./Requirements_setup.sh

    # Data upload (Download dataset from a trusted source (or use the link provided in folder "Training data") and/or use self-created dataset)
        pip install jupyter
        jupyter notebook "./Data Upload/data_upload.ipynb"
    
    # Training model script -> "./Model generation and evaluation/script.py"

    # Execute model for -
      
        Semantic transform of reviews -> Sentiment scoring of transformed reviews -> Generate restaurant rankings
          
        jupyter notebook "./Model generation and evaluation/app.ipynb"
        
# Future Scope

    1-Multi-lingual review support
    2-Dynamic real-time web scraping and updating rankings
    3-Feature extraction: ambience, food quality, service, price tags
    4-Deployment as a live recommendation app

# Acknowledgements
Inspired by personal experiences of struggling to find trustworthy dining recommendations—especially while travelling. This project aims to make restaurant choice simple, reliable, and data-driven.

# Contributions
Contributions, issues, and feature requests are welcome!

Feel free to submit a pull request.
