

# Sentiment Analysis with DistilBERT and TensorFlow

This project is a simple implementation of sentiment analysis using the DistilBERT Model on the HuggingFace Twitter dataset. The code was developed as an exercise using TensorFlow.

## Dataset

The dataset used for this project is the *__Emotion Dataset from Hugging Face__*, which consists ot Twitter messages labeld with six basic emotions: *__joy, sadness, anger, fear, love, surprise__*.

Given a Tweet, our model will be classify it into one of these emotions.

The dataset is split into the following sets:

- Training Set: 16000 labeld Tweets
- Validation Set: 2000 labeled Tweets
- Test Set: 2000 labeled Tweets

For mor information refer to the *[Hugging Face dataset page](https://huggingface.co/datasets)*.


## Brief DistilBERT Overview

DistilBERT is a smaller, distilled version of BERT   *(Bidirectional Encoder Representations from Transformers)* model. It retains much of BERT's performance while being significantly faster and requiring less memory. DistilBERT is trained using a technique called distillation, where a larger pre-trained model like BERT is used to transfer knowledge to a smaller model like DistilBERT. 

## Usage

The notebook demonstrates how to perform sentiment analysis with DistilBERT.

The model can be easily adapted to other text data or customizable parameters. 


# Suggestions and Improvements

Suggestions for improvements are welcome! If you have ideas or suggestions to enhance the project, please feel free to fork the repository and submit a pull request with your proposed changes.


# Licence 

This project is licensed under the MIT License.