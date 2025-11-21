# TasteNet — Restaurant Ranker
A deep learning and NLP-powered framework that ranks restaurants intelligently by combining numeric ratings and sentiment-analyzed text reviews. Instead of relying solely on rating averages, TasteNet predicts sentiment-based ratings, handles linguistic nuances, and computes a confidence-based ranking score for fair and reliable comparisons.

# Project Motivation
Choosing a good restaurant—especially while traveling or for special events—can be frustrating when ratings alone are misleading. Reviews may be emotional, sarcastic, biased, or inconsistent. TasteNet solves this by interpreting subjective reviews accurately and converting them into meaningful ranking signals.

# Thought process behind the scenes
    Example -
    
    "The restaurant had a good ambience but the food was a disaster"

        1 - Semantic transformation gives couple of transformations and the best one might just be -
            "The restaurant had a good ambience but the food was terrible"

        2 - Sentiment analysis classifies his transformed review on a scale of 5

            Now, I am using Bi-GRU because -
                Example: "The restaurant had a good ambience but the food was terrible"

                Forward only may focus on “good”
                Backward adds context from “terrible”
                Combined understanding improves final sentiment decision

            Use of attention because -
                Not every word contributes to sentiment. Attention assigns higher weights to emotionally relevant words.

                Attention might weight:
                Word	             Importance
                good				    0.6
                terrible			    0.9
                the, had, a, was		~0

            Both in combination handles long-range dependencies better

            GRU already solves some vanishing gradient issues, but attention ensures:
            It does not forget important earlier words
            It maintains performance even with long sentences

        Training data will have reviews which will have text review and/or numeric ratings

        3 - In case numeric ratings are absent, for example- opinions of people on social media like reddit, etc., these numeric ratings are predicted from sentiment scoring explained above.
            For reviews where numeric ratings are present, they are taken as it is

        Now, all numeric ratings corresponding to each restaurant is taken

        4 - Why confidence score depicts more reliable rankings ?
            Example:
            Restaurant	    Avg Rating	    Review Count	Rating Based On Spread	    Final Ranking
            A	            4.2	            180	            	0.82  (Narrow)	            High
            B	            4.2	            32	            	0.54  (Wide)	            Lower
            
            A high spread means users have polarising experiences (good + terrible).
            A narrow spread means predictable quality.
            
            Consumers typically prefer consistency, so the ranking should reflect stability, not only the arithmetic mean.

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
