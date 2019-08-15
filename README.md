# Sentiment-Analysis
Text Classification by using Convolutional Neural Network (CNN) algorithm

## Goal of present work
&emsp;&emsp; To automatically classify the customer comments of the most famous hospitals (Ramathibodi, Siriraj, and Chulalongkorn hospital) in Thailand into 2 categories as positive and negative issues by using CNN algorithm.

## Convolutional Neural Network (CNN)
&emsp;&emsp; [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) is a class of deep neural networks which is usually applied to image classification but recent [work](https://arxiv.org/pdf/1408.5882.pdf) applied CNN for text classification which obtained reliable prediction. 

&emsp;&emsp; Principle concept of CNN for natural language processing (NLP) is using of sliding filter window over vectorized features. Basically, a customer comment consists of one or more sentences which commonly be converted by word vectorization resulting as 1-dimensional vectorized features. Therefore, width of the filter is commonly equal to 1 due to the 1-dimension whereas height is vary between 2-5 which is similar to [n-grams](https://en.wikipedia.org/wiki/N-gram) modeling in NLP. For example, if stride is equal to 1 and filter (kernel) size is 1x2, sliding window of CNN algorithm will act as bi-gram modeling in NLP. Moreover, advantages of using CNN for text classification are: _1) fast learning compared to other algorithm; and 2) a small number of hyperparameters used for tuning.

&emsp;&emsp; The present work consists of 4 sections which are uploaded as .ipynb files.

### Part I: Data Retrieval <br>
&emsp;&emsp; Patient comments were retrieved from [HonestDocs](https://www.honestdocs.co). The data included only top 3 well-known hospitals in Thailand such as Ramathibodi, Siriraj, and Chulalonkorn hospital. Then the comments were translated from Thai to English language by using [Translation Client Libraries for the Cloud Translation API of Google APIs](https://cloud.google.com/translate/docs/quickstart-client-libraries)

### Part II: Data Preprocessing <br>
&emsp;&emsp; The translated comments were splitted into training, validation, and test data set. Then all dataset were cleaned and transformed to vectorized features.

### Part III: Modeling & Evaluation <br>
&emsp;&emsp; Model learning by given training data set and tuning hyperparameters by given validation data set. Then model performance evaluation and error analysis are done by given test data set.

### Part IV: Sentiment analysis and Suggestion <br>
&emsp;&emsp; Apply the optimum model to classify whole data set of the patient comments into positive and negative issues. Then analyse and discuss with colleagues who built other models to give appropriate suggestion for the hospitals.

<br></br>
__Contributors:__ [ponthongmak](https://github.com/ponthongmak) , [petchpanu](https://github.com/petchpanu), [perlestot](https://github.com/perlestot)
