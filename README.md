# Machine-Learning
Various machine learning algorithms broken down in basic and readable python code. Useful for studying and learning how the algorithms function.

* popcorn.sentiment.sklearn.py - basic sentiment analysis script that converts list of phrases into bag of words features and does logistic regression. Achieved [kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard) score of .962 and 8th place at the time of writing. 

* LinearRegression.py - python version of simple linear regression from scratch!

* LogisticRegression.py - gradient descent logistic regression. Currently gets 60% on admission dataset whereas R's default logistic regression gets 70%.

* BackPropagationNN.py - Basic MultiLayer Perceptron (MLP) network, adapted and from the book ['Programming Collective Intelligence'](http://shop.oreilly.com/product/9780596529321.do) Consists of three layers: input, hidden and output. The sizes of input and output must match data the size of hidden is user defined when initializing the network. The algorithm has been generalized to be used on any dataset. Write up on my [blog](http://databoys.github.io/Feedforward/)

* MultiLayerPerceptron.py - Essentially the same as 'BackPropagationNN.py' except optimized with numpy to increase the speed. 