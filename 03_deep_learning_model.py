from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

# LOAD DATA: start
def load_doc(filename):
    """
    param:  filename
    return: text

    Load file content into memory.
    """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_set(filename):
    """
    param:  filename
    return: dataset

    Load the list of photo identifiers which has been predefined in:
        Flickr8k_text/Flickr_8k.devImages.txt
    """
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1: # skip empty lines
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
    """
    param:  filename, dataset
    return: descriptions

    Load the clean descriptions contained in the previously generated descriptions.txt.
    Then create a list using these descriptions.
    """
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_photo_features(filename, dataset):
    """
    param:  filename, dataset
    return: features

    Load the entire set of photo descriptions.
    Return the subset for a given set of photo descriptions.
    """
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}   # filter features
    return features
# LOAD DATA: end

# ENCODING: start
def to_lines(descriptions):
    """
    param:  descriptions
    return: all_desc

    Conversion of a clean description dictionary to a list of descriptions.
    """
    all_desc = list()
    for key in descriptions.keys():
    	[all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    """
    param:  descriptions
    return: tokenizer

    Fit a tokenizer given the loaded photo description text
    """
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    """
    param:  descriptions
    return: length of the description with the most words

    Run through all the descriptions.
    Return the longest.
    """
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def create_sequences(tokenizer, max_length, descriptions, photos):
    """
    params: tokenzier, max_length, descriptions, Photos
    return: photo id, input sequence, output sequence

    X1 and X2 are the inputs to the model.
    1: photo features
    2: encoded text (integer) : fed to a word embedding layer.
    y is the output which is the encoded next word in the text sequence.
    y: prediction -> probability distribution over all words in the vocabulary.
    """
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
    	for desc in desc_list:
    		seq = tokenizer.texts_to_sequences([desc])[0] # encode the sequence
    		for i in range(1, len(seq)):
    			in_seq, out_seq = seq[:i], seq[i]
    			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]  # pad input sequence
    			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]  # encode output sequence
    			X1.append(photos[key][0])
    			X2.append(in_seq)
    			y.append(out_seq)
    return array(X1), array(X2), array(y)
# ENCODING: end

# DEFINING THE MODEL (MERGE-MODEL): start
def define_model(vocab_size, max_length):
    """
    params: vocab_size, max_length
    return: model
    Define the captioning model.

    => 256 elements vectors: this model configuration learns very fast

    1.
    16-layer VGG model pre-trained on the ImageNet dataset.
    The photos have been pre-processed with the VGG model (without the output layer)
    Input: extracted features predicted by this model
    -> vector of 4096 elements
    -> Dense layer: 256 elements representation of the photo

    2.
    Word embedding layer which handles the text input +
    Long Short-Term Memory (LSTM) recurrent neural network layer.
    -> expectes a pre-defined length: 34 words
    -> these are then fed into an Embedding layer (that ises a mask to ignore padded values).
    -> LSMT layer: 256 memory units

    3.
    Feature extractor + sequence processor outputs => fiexed length vector.
    Final prediction.
    -> input models are merged (addition operation)
    -> fed to a Dense 256 neuron layer
    -> final output: softmax prediction over the entire output vocabulary for the next word in sequence.
    """
    # 1. feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # 2. sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # 3. decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
# DEFINING THE MODEL (MERGE-MODEL): end

if __name__ == "__main__":
    # train dataset

    # load training dataset (6K)
    filename = '../Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))
    # photo features
    train_features = load_photo_features('features.pkl', train)
    print('Photos: train=%d' % len(train_features))
    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)
    # prepare sequences
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

    # dev dataset

    # load test set
    filename = '../Flickr8k_text/Flickr_8k.devImages.txt'
    test = load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = load_clean_descriptions('descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = load_photo_features('features.pkl', test)
    print('Photos: test=%d' % len(test_features))
    # prepare sequences
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

    # fit model

    # define the model
    model = define_model(vocab_size, max_length)
    # define checkpoint callback
    filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # fit model
    model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
