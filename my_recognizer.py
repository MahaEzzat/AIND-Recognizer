import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    for word_id in test_set.get_all_sequences().keys():
        X_word,lengths_word=test_set.get_all_Xlengths()[word_id]
        word_probability = {}
        for word in models.keys():
            model = models[word]
            try:
                LogLvalue = round(model.score(X_word,lengths_word),3)
            except:
                LogLvalue=float("-inf")
            word_probability[word] = LogLvalue
        probabilities.append(word_probability)
        max_probability = max(word_probability.values())
        guesses_words = [word for word in word_probability.keys() if word_probability[word]==max_probability] 
        guesses.append(guesses_words[0])
    
    return (probabilities,guesses)
