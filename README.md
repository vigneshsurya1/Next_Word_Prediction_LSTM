# Next Word Prediction using LSTM

## Project Overview
This project focuses on **next word prediction** using the text from Shakespeare's *Hamlet*. The goal is to predict the most probable next word in a sequence based on the input text. The model is implemented using a **Long Short-Term Memory (LSTM)** neural network, a powerful tool for sequence prediction tasks in natural language processing (NLP).

---

## Dataset
- **Name**: Shakespeare's *Hamlet*
- **Description**: The text of the play *Hamlet* is used as the dataset, preprocessed for sequence modeling.
- **Usage**: The text is split into sequences to train the LSTM model for predicting the next word.

---

## Techniques
- **Text Preprocessing**:
  - Tokenization and text cleaning using NLTK.
  - Creating input-output pairs for sequence modeling.
  - Saving preprocessed data using Pickle.
- **Model Architecture**:
  - LSTM implemented with TensorFlow and Keras.
  - Sequential architecture with embedding and dense layers.

---

## Libraries Used
- **TensorFlow**: For building and training the LSTM model.
- **Keras**: High-level API for defining and training the model.
- **NumPy**: For numerical computations.
- **Pandas**: For dataset manipulation.
- **NLTK**: For text preprocessing (e.g., tokenization).
- **Pickle**: For saving and loading preprocessed data.
- **Streamlit**: For deploying the model and generating predictions interactively.

---

## Results
- **Accuracy Score**: 0.6781
- The model predicts the next word with a reasonable level of accuracy, considering the complexity of the language.

---

## How to Run
1. Clone the repository or download the notebook file.
2. Install the required libraries:
   ```bash
   pip install tensorflow keras numpy pandas nltk pickle streamlit
   ```
3. Run the notebook to preprocess the data, train the model, and evaluate its performance.
4. Deploy the model using Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## Future Improvements
- Experiment with larger datasets or pre-trained embeddings like GloVe.
- Optimize hyperparameters to improve accuracy.
- Extend the application to support multiple text genres or authors.

---

## Acknowledgments
- **Dataset**: Text from Shakespeare's *Hamlet*.
- **Libraries**: TensorFlow and Keras for deep learning implementation.

