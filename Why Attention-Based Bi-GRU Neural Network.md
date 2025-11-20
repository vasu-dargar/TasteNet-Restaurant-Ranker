# What advantage does Bi-GRU with attention give on sentiment analysis?
Using Bi-GRU with Attention in sentiment analysis provides several important advantages over basic RNN/LSTM/GRU or simple classifiers.


1 - Understands context from both directions

A Bidirectional GRU reads text forward and backward, allowing the model to understand meaning based on words before and after a token.

    Example: "The movie isn’t good but the ending was amazing."
    
    Forward only may focus on “isn’t good”
    Backward adds context from “amazing”
    Combined understanding improves final sentiment decision


2 - Focuses on important words via Attention

Not every word contributes to sentiment. Attention assigns higher weights to emotionally relevant words.

    Example: "The laptop is lightweight, fast, and absolutely fantastic."

    Attention might weight:
    Word	                    Importance
    lightweight	                0.3
    fast	                    0.2
    fantastic	                0.9
    is, the, and, absolutely	~0

So the model focuses on important sentiment-carrying words, improving accuracy and explainability.


3 - Handles long-range dependencies better

    GRU already solves some vanishing gradient issues, but attention ensures:
    It does not forget important earlier words
    It maintains performance even with long sentences

This is crucial in reviews like:

"Although the first half was boring, the second half completely changed everything."


4 - Computationally efficient vs Bi-LSTM

GRU has fewer parameters → faster training & inference

Better suited for real-time sentiment applications like chat, reviews, customer feedback bots


5 - Proven higher performance

Many NLP benchmarks show Bi-GRU + Attention > GRU/LSTM > traditional ML models like SVM, Naive Bayes.

    Typical improvements:
    
    Model	            Accuracy Example
    Bi-GRU + Attention	91–95%
    GRU	                88–90%
    LSTM	            86–89%
    SVM TF-IDF	        80–85%
    
    (Actual results differ by dataset)


# Conclusion
Attention-Based Bi-GRU Neural Network is used to overcome problems in deep learning algorithms such as LSTM and GRU, which cannot capture important information in sequence learning. Hence, improving Sentiment Classification of restaurant reviews with use of bidirectional GRU by combining global attention and word2vec as word embedding. Global attention works by focusing on the most contributing words so that they form like keywords. The results of the experiment are satisfactory.


# Future scope

    The suggestions for further experiments are:
    
    1-Use other attention mechanisms such as self-attention, multi-head, and hierarchical attention to improve performance.
    2-Implement cross-validation testing during training to reduce overfitting and underfitting.
    3-Using other word embeddings such as GloVe and FastText.
    4-Apply the proposed model to an actual application
