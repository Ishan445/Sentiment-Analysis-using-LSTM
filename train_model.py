{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "65a2fcf5-21ec-47f0-abcf-39bafed904d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in c:\\users\\izadb\\anaconda3\\lib\\site-packages (2.20.0)\n",
      "Requirement already satisfied: pandas in c:\\users\\izadb\\anaconda3\\lib\\site-packages (2.2.2)\n",
      "Requirement already satisfied: numpy in c:\\users\\izadb\\anaconda3\\lib\\site-packages (1.26.4)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\izadb\\anaconda3\\lib\\site-packages (1.5.1)\n",
      "Requirement already satisfied: beautifulsoup4 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (4.12.3)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (2.3.1)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=24.3.25 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (25.9.23)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (0.6.0)\n",
      "Requirement already satisfied: google_pasta>=0.1.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (0.2.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (18.1.1)\n",
      "Requirement already satisfied: opt_einsum>=2.3.2 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (3.4.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (24.1)\n",
      "Requirement already satisfied: protobuf>=5.28.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (5.29.5)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (2.32.3)\n",
      "Requirement already satisfied: setuptools in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (75.1.0)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (3.1.0)\n",
      "Requirement already satisfied: typing_extensions>=3.6.6 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (4.15.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (1.14.1)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (1.76.0)\n",
      "Requirement already satisfied: tensorboard~=2.20.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (2.20.0)\n",
      "Requirement already satisfied: keras>=3.10.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (3.11.3)\n",
      "Requirement already satisfied: h5py>=3.11.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (3.11.0)\n",
      "Requirement already satisfied: ml_dtypes<1.0.0,>=0.5.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorflow) (0.5.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: scipy>=1.6.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from scikit-learn) (1.13.1)\n",
      "Requirement already satisfied: joblib>=1.2.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from scikit-learn) (1.4.2)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from scikit-learn) (3.5.0)\n",
      "Requirement already satisfied: soupsieve>1.2 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from beautifulsoup4) (2.5)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)\n",
      "Requirement already satisfied: rich in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from keras>=3.10.0->tensorflow) (13.7.1)\n",
      "Requirement already satisfied: namex in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from keras>=3.10.0->tensorflow) (0.1.0)\n",
      "Requirement already satisfied: optree in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from keras>=3.10.0->tensorflow) (0.17.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow) (2025.8.3)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorboard~=2.20.0->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: pillow in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorboard~=2.20.0->tensorflow) (10.4.0)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorboard~=2.20.0->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from tensorboard~=2.20.0->tensorflow) (3.0.3)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard~=2.20.0->tensorflow) (2.1.3)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from rich->keras>=3.10.0->tensorflow) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from rich->keras>=3.10.0->tensorflow) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\izadb\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.10.0->tensorflow) (0.1.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install tensorflow pandas numpy scikit-learn beautifulsoup4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "75fd09b9-95c5-47e1-a4dc-e009f54d5d93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting data load... this might take a minute.\n",
      "Loading pos reviews from C:\\Users\\izadb\\Downloads\\aclImdb_v1\\aclImdb\\train\\pos...\n",
      "Loading neg reviews from C:\\Users\\izadb\\Downloads\\aclImdb_v1\\aclImdb\\train\\neg...\n",
      "Successfully loaded 25000 training reviews.\n",
      "Loading pos reviews from C:\\Users\\izadb\\Downloads\\aclImdb_v1\\aclImdb\\test\\pos...\n",
      "Loading neg reviews from C:\\Users\\izadb\\Downloads\\aclImdb_v1\\aclImdb\\test\\neg...\n",
      "Successfully loaded 25000 testing reviews.\n",
      "\n",
      "Sample Data:\n",
      "                                              review  sentiment\n",
      "0  Bromwell High is a cartoon comedy. It ran at t...          1\n",
      "1  Homelessness (or Houselessness as George Carli...          1\n",
      "2  Brilliant over-acting by Lesley Ann Warren. Be...          1\n",
      "3  This is easily the most underrated film inn th...          1\n",
      "4  This is not the typical Mel Brooks film. It wa...          1\n"
     ]
    }
   ],
   "source": [
    "import os \n",
    "import pandas as pd\n",
    "\n",
    "# 1. Define the path to your extracted folder\n",
    "# If your python file is next to the 'train' folder, this works.\n",
    "# Otherwise, put the full path like: \"C:/Users/You/Downloads/aclImdb/train\"\n",
    "train_dir = r\"C:\\Users\\izadb\\Downloads\\aclImdb_v1\\aclImdb\\train\" \n",
    "test_dir = r\"C:\\Users\\izadb\\Downloads\\aclImdb_v1\\aclImdb\\test\"\n",
    "\n",
    "def load_data_from_folder(directory):\n",
    "    data_list = []\n",
    "    \n",
    "    # We want to look inside 'pos' (positive) and 'neg' (negative) subfolders\n",
    "    for label in ['pos', 'neg']:\n",
    "        path = os.path.join(directory, label)\n",
    "        \n",
    "        # Check if the folder exists to avoid errors\n",
    "        if not os.path.exists(path):\n",
    "            print(f\"Warning: Folder {path} not found!\")\n",
    "            continue\n",
    "            \n",
    "        # Loop through every file in that folder\n",
    "        print(f\"Loading {label} reviews from {path}...\")\n",
    "        for filename in os.listdir(path):\n",
    "            if filename.endswith('.txt'):\n",
    "                file_path = os.path.join(path, filename)\n",
    "                \n",
    "                # Read the text content of the file\n",
    "                with open(file_path, encoding='utf-8') as f:\n",
    "                    review_text = f.read()\n",
    "                \n",
    "                # Assign sentiment: 1 for positive, 0 for negative\n",
    "                sentiment_score = 1 if label == 'pos' else 0\n",
    "                \n",
    "                # Add to our list\n",
    "                data_list.append({'review': review_text, 'sentiment': sentiment_score})\n",
    "                \n",
    "    # Convert list of dictionaries to a Pandas DataFrame (Table)\n",
    "    return pd.DataFrame(data_list)\n",
    "\n",
    "# --- EXECUTE THE LOADING ---\n",
    "print(\"Starting data load... this might take a minute.\")\n",
    "\n",
    "# Load Training Data\n",
    "train_df = load_data_from_folder(train_dir)\n",
    "print(f\"Successfully loaded {len(train_df)} training reviews.\")\n",
    "\n",
    "# Load Test Data\n",
    "test_df = load_data_from_folder(test_dir)\n",
    "print(f\"Successfully loaded {len(test_df)} testing reviews.\")\n",
    "\n",
    "# Show us the top 5 rows to verify\n",
    "print(\"\\nSample Data:\")\n",
    "print(train_df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b32d9284-ea86-455d-9f8d-02f80036f64f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cleaning text...\n",
      "Fitting tokenizer...\n",
      "Vocabulary size: 88583\n",
      "\n",
      "--- Data Ready for Training ---\n",
      "Training Matrix Shape: (25000, 250)\n",
      "Testing Matrix Shape: (25000, 250)\n",
      "\n",
      "Original Text (First 50 chars):\n",
      "bromwell high is a cartoon comedy. it ran at the s\n",
      "\n",
      "Converted to Sequence:\n",
      "[   1  309    7    4 1069  209    9 2161   30    2  169   55   14   46\n",
      "   82 5844   41  392  110  138   14 5340   58 4449  150    8    2 4988\n",
      " 5924  482   69    6  261   12    1    1 2002    7   73 2425    6  632\n",
      "   71    7 5340    2    1    6 2003    1    2 5925 1534   34   67   64\n",
      "  205  140   65 1230    1    1    2    1    5    2  223  901   29 3022\n",
      "   69    5    2 5845   10  693    3   65 1534   51   10  216    2  387\n",
      "    8   60    4 1467 3712  800    6 3513  177    2  392   10 1237    1\n",
      "   30  309    4  353  344 2974  143  130    6 7799   28    5  126 5340\n",
      " 1467 2373    6    1  309   10  532   12  108 1468    5   58  555  101\n",
      "   12    1  309    7  227 4174   48    4 2232   12    9  215    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0    0    0\n",
      "    0    0    0    0    0    0    0    0    0    0    0    0]\n"
     ]
    }
   ],
   "source": [
    "#step 2\n",
    "\n",
    "import re\n",
    "import numpy as np\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "\n",
    "# --- HYPERPARAMETERS ---\n",
    "# We will only look at the top 10,000 most frequent words.\n",
    "# Rare words will be replaced by a special \"OOV\" (Out Of Vocabulary) token.\n",
    "VOCAB_SIZE = 10000 \n",
    "# We will cut off reviews after 250 words or pad shorter ones to reach 250.\n",
    "MAX_LENGTH = 250 \n",
    "OOV_TOK = \"<OOV>\"\n",
    "\n",
    "def clean_text(text):\n",
    "    \"\"\"\n",
    "    Removes HTML tags and special characters.\n",
    "    \"\"\"\n",
    "    # Remove HTML tags (like <br />) using Regex\n",
    "    text = re.sub(r'<br\\s*/?>', ' ', text)\n",
    "    \n",
    "    # Optional: Remove non-alphabetic characters (keep only words)\n",
    "    # text = re.sub(r'[^a-zA-Z\\s]', '', text)\n",
    "    \n",
    "    return text.lower()\n",
    "\n",
    "# 1. Apply Cleaning\n",
    "print(\"Cleaning text...\")\n",
    "train_df['clean_review'] = train_df['review'].apply(clean_text)\n",
    "test_df['clean_review'] = test_df['review'].apply(clean_text)\n",
    "\n",
    "# 2. Fit Tokenizer (Learn the Vocabulary)\n",
    "# IMPORTANT: We only fit on TRAINING data. The test data must remain \"unseen\".\n",
    "print(\"Fitting tokenizer...\")\n",
    "tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)\n",
    "tokenizer.fit_on_texts(train_df['clean_review'])\n",
    "\n",
    "# Get the word index (dictionary of word -> number)\n",
    "word_index = tokenizer.word_index\n",
    "print(f\"Vocabulary size: {len(word_index)}\")\n",
    "\n",
    "# 3. Convert Text to Sequences of Numbers\n",
    "train_sequences = tokenizer.texts_to_sequences(train_df['clean_review'])\n",
    "test_sequences = tokenizer.texts_to_sequences(test_df['clean_review'])\n",
    "\n",
    "# 4. Pad Sequences to ensure uniform length\n",
    "# padding='post' adds zeros at the end. truncating='post' cuts off the end.\n",
    "X_train = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')\n",
    "X_test = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')\n",
    "\n",
    "# Prepare Labels\n",
    "y_train = np.array(train_df['sentiment'])\n",
    "y_test = np.array(test_df['sentiment'])\n",
    "\n",
    "print(\"\\n--- Data Ready for Training ---\")\n",
    "print(f\"Training Matrix Shape: {X_train.shape}\")\n",
    "print(f\"Testing Matrix Shape: {X_test.shape}\")\n",
    "\n",
    "# Let's see what a review looks like now\n",
    "print(\"\\nOriginal Text (First 50 chars):\")\n",
    "print(train_df['clean_review'][0][:50])\n",
    "print(\"\\nConverted to Sequence:\")\n",
    "print(X_train[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "be77f845-8beb-4a49-8143-c4d0a29b2317",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\izadb\\anaconda3\\Lib\\site-packages\\keras\\src\\layers\\core\\embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential_1\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential_1\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ embedding_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Embedding</span>)         │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ bidirectional_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Bidirectional</span>) │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ ?                      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ embedding_1 (\u001b[38;5;33mEmbedding\u001b[0m)         │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ bidirectional_1 (\u001b[38;5;33mBidirectional\u001b[0m) │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_2 (\u001b[38;5;33mDense\u001b[0m)                 │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)             │ ?                      │             \u001b[38;5;34m0\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_3 (\u001b[38;5;33mDense\u001b[0m)                 │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "None\n"
     ]
    }
   ],
   "source": [
    "# step 3\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout\n",
    "\n",
    "# --- MODEL HYPERPARAMETERS ---\n",
    "EMBEDDING_DIM = 64  # Size of the vector representation for each word\n",
    "LSTM_UNITS = 64     # Number of \"memory units\" in the LSTM layer\n",
    "\n",
    "def build_model():\n",
    "    model = Sequential([\n",
    "        # Layer 1: Embedding\n",
    "        # Converts integer sequences (e.g., [4, 25, ...]) into dense vectors.\n",
    "        # input_dim=VOCAB_SIZE (10000), output_dim=64, input_length=MAX_LENGTH (250)\n",
    "        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),\n",
    "        \n",
    "        # Layer 2: Bidirectional LSTM\n",
    "        # We wrap LSTM in Bidirectional() to read text both ways.\n",
    "        # dropout=0.2 helps prevent overfitting (randomly ignoring 20% of neurons during training)\n",
    "        Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, return_sequences=False)),\n",
    "        \n",
    "        # Layer 3: Dense Output\n",
    "        # A single neuron with 'sigmoid' activation outputs a value between 0 and 1.\n",
    "        # 0 = Negative, 1 = Positive\n",
    "        Dense(64, activation='relu'), # Optional intermediate dense layer for better feature extraction\n",
    "        Dropout(0.5),                 # Extra dropout to be safe\n",
    "        Dense(1, activation='sigmoid')\n",
    "    ])\n",
    "    return model\n",
    "\n",
    "# Build the model\n",
    "model = build_model()\n",
    "\n",
    "# Compile the model\n",
    "# Optimizer: Adam (standard for NLP)\n",
    "# Loss: Binary Crossentropy (standard for Yes/No classification)\n",
    "# Metrics: Accuracy\n",
    "model.compile(optimizer='adam',\n",
    "              loss='binary_crossentropy',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "# View the architecture\n",
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "55f5aa34-04d0-4f88-ab83-fc3f854f9a23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting training with safeguards...\n",
      "Epoch 1/10\n",
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1s/step - accuracy: 0.6211 - loss: 0.6518  \n",
      "Epoch 1: val_accuracy improved from None to 0.65140, saving model to best_model.h5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m197s\u001b[0m 1s/step - accuracy: 0.6772 - loss: 0.5935 - val_accuracy: 0.6514 - val_loss: 0.6533\n",
      "Epoch 2/10\n",
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1s/step - accuracy: 0.8443 - loss: 0.3795  \n",
      "Epoch 2: val_accuracy improved from 0.65140 to 0.76460, saving model to best_model.h5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m187s\u001b[0m 1s/step - accuracy: 0.8587 - loss: 0.3514 - val_accuracy: 0.7646 - val_loss: 0.5134\n",
      "Epoch 3/10\n",
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1s/step - accuracy: 0.9159 - loss: 0.2323  \n",
      "Epoch 3: val_accuracy improved from 0.76460 to 0.77120, saving model to best_model.h5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m190s\u001b[0m 1s/step - accuracy: 0.9095 - loss: 0.2413 - val_accuracy: 0.7712 - val_loss: 0.5057\n",
      "Epoch 4/10\n",
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1s/step - accuracy: 0.9366 - loss: 0.1811  \n",
      "Epoch 4: val_accuracy improved from 0.77120 to 0.81100, saving model to best_model.h5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m196s\u001b[0m 1s/step - accuracy: 0.9345 - loss: 0.1851 - val_accuracy: 0.8110 - val_loss: 0.4977\n",
      "Epoch 5/10\n",
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1s/step - accuracy: 0.9529 - loss: 0.1429  \n",
      "Epoch 5: val_accuracy did not improve from 0.81100\n",
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m194s\u001b[0m 1s/step - accuracy: 0.9413 - loss: 0.1657 - val_accuracy: 0.7712 - val_loss: 0.6508\n",
      "Epoch 6/10\n",
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1s/step - accuracy: 0.9553 - loss: 0.1287  \n",
      "Epoch 6: val_accuracy improved from 0.81100 to 0.81540, saving model to best_model.h5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m191s\u001b[0m 1s/step - accuracy: 0.9529 - loss: 0.1325 - val_accuracy: 0.8154 - val_loss: 0.6036\n",
      "\n",
      "Saved the best model to 'best_model.h5' and tokenizer to 'tokenizer.pickle'\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "\n",
    "# 1. Define Callbacks\n",
    "# Stop if validation loss doesn't improve for 2 epochs (patience=2)\n",
    "early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)\n",
    "\n",
    "# Always save the best version of the model to a specific file\n",
    "checkpoint = ModelCheckpoint('best_model.h5', \n",
    "                             monitor='val_accuracy', \n",
    "                             save_best_only=True, \n",
    "                             mode='max', \n",
    "                             verbose=1)\n",
    "\n",
    "print(f\"Starting training with safeguards...\")\n",
    "\n",
    "# 2. Train with Callbacks\n",
    "history = model.fit(X_train, y_train,\n",
    "                    batch_size=BATCH_SIZE,\n",
    "                    epochs=10,  # We can set this high now because EarlyStopping will cut it short!\n",
    "                    validation_split=0.2,\n",
    "                    callbacks=[early_stop, checkpoint], # <--- Add them here\n",
    "                    verbose=1)\n",
    "\n",
    "# 3. Save the final tokenizer (doesn't change during training)\n",
    "import pickle\n",
    "with open('tokenizer.pickle', 'wb') as handle:\n",
    "    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "\n",
    "print(\"\\nSaved the best model to 'best_model.h5' and tokenizer to 'tokenizer.pickle'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95af3172-7484-46bc-bc5f-6693ccc5b19b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
