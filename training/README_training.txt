==========================================================
Training Module - Usage Guide
==========================================================

This folder contains the training pipeline of the project.

Its purpose is to allow retraining of the leakage classifier
on a new dataset in the future, without changing the main
PDF analysis system.

----------------------------------------------------------
FILES
----------------------------------------------------------

1. train_model.py
   Trains a new leakage classifier using a dataset located at:
   data/training_dataset.csv

2. prepare_dataset.py
   Helps convert an external dataset into the required format
   used by train_model.py

----------------------------------------------------------
EXPECTED DATASET FORMAT
----------------------------------------------------------

The training dataset must be a CSV file with the following columns:

- text
- label

Where:
- text  = the sentence or text segment
- label = numeric class
          0 = NON_LEAK
          1 = LEAK

Example:

text,label
"Hello, how are you?",0
"My phone number is 0501234567",1
"The team completed the sprint",0
"The customer's ID number is 123456789",1

----------------------------------------------------------
HOW TO TRAIN A NEW MODEL
----------------------------------------------------------

Step 1:
Place your dataset in:
data/training_dataset.csv

Step 2:
Run the training script:

python training/train_model.py

Step 3:
After successful training, the following files will be updated:

models/leak_classifier.pkl
models/tfidf_vectorizer.pkl

The main system will automatically use the new model files.

----------------------------------------------------------
HOW TO PREPARE A DIFFERENT DATASET FORMAT
----------------------------------------------------------

If your new dataset has different column names, for example:

sentence,class_name

then:

1. Place it in:
   data/raw_dataset.csv

2. Open:
   training/prepare_dataset.py

3. Update the following values:
   TEXT_COLUMN_NAME
   LABEL_COLUMN_NAME
   LABEL_MAPPING

4. Run:
   python training/prepare_dataset.py

This will create:
data/training_dataset.csv

Then run:
python training/train_model.py

----------------------------------------------------------
NOTES
----------------------------------------------------------

- The current system supports binary classification only:
  LEAK / NON_LEAK

- If you want multi-class classification in the future,
  the training script and inference logic must be updated.

- Retraining the model does not require changes to main.py,
  as long as the new model is saved to the same paths.