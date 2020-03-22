#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Group 4
# Alena Clim: 2013151
# Maartje van Berkel: 2010960
# Merijn Broos: 2010284

# We used a simple interpolation method for a bigram model.
# I wasn't able to make it work generally, but it works for the specific value of n=2. 


# In[2]:


import json
import numpy as np
from collections import Counter, defaultdict


# In[39]:


# I think these blocks of code shouldn't have an influence on the way you'll test our code,
# because you'll already have two separate files.
# But I decided to let it here, so you can see the way we split the data

# Opening a json file. The training set you gave us, I called it corpus and then split it
with open('/_MYstuff/Desktop/Uni/Computational Linguistics/corpus.json') as f:
    data = json.load(f)

# I am checking the length    
length = len(data)
print("Length corpus ", length)

# Splitting a list of lists with the 80/20 percentage for train/test sets 
train = data[:int(length*0.8)]
test = data[int(length*0.8):]

# Here I waanted to make sure that I'm not missing elements, added they should be the previously printed length
print("Length train ", len(train))
print("Length test ", len(test))

# This was the method I found to create two new json files.
# The only problem is that if this piece of code is run multiple times it gives an error 
# because it's trying to create the files over and over again
with open('training.json', 'x') as outfile:
    json.dump(train, outfile) 

with open('test.json', 'x') as outfile:
    json.dump(test, outfile) 


# In[104]:


# These were my paths after forming two new files from the original corpus

train_path = '/_MYstuff/Desktop/Uni/Computational Linguistics/training.json'
test_path = '/_MYstuff/Desktop/Uni/Computational Linguistics/test.json'


# In[37]:


# I haven't changed anything here

class Corpus(object):
    
    """
    This class creates a corpus object read off a .json file consisting of a list of lists,
    where each inner list is a sentence encoded as a list of strings.
    """
    
    def __init__(self, path, t, n, bos_eos=True, vocab=None):
        
        """
        DON'T TOUCH THIS CLASS! 
        IT'S HERE TO SHOW THE PROCESS, YOU DON'T NEED TO ANYTHING HERE. 
        
        A Corpus object has the following attributes:
         - vocab: set or None (default). If a set is passed, words in the input file not 
                         found in the set are replaced with the UNK string
         - path: str, the path to the .json file used to build the corpus object
         - t: int, words with frequency count < t are replaced with the UNK string
         - ngram_size: int, 2 for bigrams, 3 for trigrams, and so on.
         - bos_eos: bool, default to True. If False, bos and eos symbols are not 
                     prepended and appended to sentences.
         - sentences: list of lists, containing the input sentences after lowercasing and 
                         splitting at the white space
         - frequencies: Counter, mapping tokens to their frequency count in the corpus
        """
        
        self.vocab = vocab        
        self.path = path
        self.t = t
        self.ngram_size = n
        self.bos_eos = bos_eos
        
        self.sentences = self.read()
        # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
    
        self.frequencies = self.freq_distr()
        # output --> Counter('the': 485099, 'of': 301877, 'i': 286549, ...)
        # the numbers are made up, they aren't the actual frequency counts
        
        if self.t or self.vocab:
            # input --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
            self.sentences = self.filter_words()
            # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'UNK', '.'], ...]
            # supposing that park wasn't frequent enough or was outside of the training 
            # vocabulary, it gets replaced by the UNK string
            
        if self.bos_eos:
            # input --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
            self.sentences = self.add_bos_eos()
            # output --> [['bos', i', 'am', 'home' '.', 'eos'], 
            #             ['bos', you', 'went', 'to', 'the', 'park', '.', 'eos'], ...]
                    
    def read(self):
        
        """
        Reads the sentences off the .json file, replaces quotes, lowercases strings and splits 
        at the white space. Returns a list of lists.
        """
        
        if self.path.endswith('.json'):
            sentences = json.load(open(self.path, 'r'))                
        else:   
            sentences = []
            with open(self.path, 'r', encoding='latin-1') as f:
                for line in f:
                    print(line[:20])
                    # first strip away newline symbols and the like, then replace ' and " with the empty 
                    # string and get rid of possible remaining trailing spaces 
                    line = line.strip().translate({ord(i): None for i in '"\'\\'}).strip(' ')
                    # lowercase and split at the white space (the corpus has ben previously tokenized)
                    sentences.append(line.lower().split(' '))
        
        return sentences
    
    def freq_distr(self):
        
        """
        Creates a counter mapping tokens to frequency counts
        
        count = Counter()
        for sentence in self.sentences:
            for word in sentence:
                count[w] += 1
            
        """
    
        return Counter([word for sentence in self.sentences for word in sentence])
        
    
    def filter_words(self):
        
        """
        Replaces illegal tokens with the UNK string. A token is illegal if its frequency count
        is lower than the given threshold and/or if it falls outside the specified vocabulary.
        The two filters can be both active at the same time but don't have to be. To exclude the 
        frequency filter, set t=0 in the class call.
        """
        
        filtered_sentences = []
        for sentence in self.sentences:
            filtered_sentence = []
            for word in sentence:
                if self.t and self.vocab:
                    # check that the word is frequent enough and occurs in the vocabulary
                    filtered_sentence.append(
                        word if self.frequencies[word] > self.t and word in self.vocab else 'UNK'
                    )
                else:
                    if self.t:
                        # check that the word is frequent enough
                        filtered_sentence.append(word if self.frequencies[word] > self.t else 'UNK')
                    else:
                        # check if the word occurs in the vocabulary
                        filtered_sentence.append(word if word in self.vocab else 'UNK')
                        
            if len(filtered_sentence) > 1:
                # make sure that the sentence contains more than 1 token
                filtered_sentences.append(filtered_sentence)
    
        return filtered_sentences
    
    def add_bos_eos(self):
        
        """
        Adds the necessary number of BOS symbols and one EOS symbol.
        
        In a bigram model, you need one bos and one eos; in a trigram model you need two bos and one eos, 
        and so on...
        """
        
        padded_sentences = []
        for sentence in self.sentences:
            padded_sentence = ['#bos#']*(self.ngram_size-1) + sentence + ['#eos#']
            padded_sentences.append(padded_sentence)
    
        return padded_sentences


# In[105]:


# In this class I changed the get_ngram_probabilities function to return an interpolated bigram model.
# I hardcored the values for lambda in the function so I didn't need to change the definition parameters.
# Right now the lam parameter is not used (since it returns the same perplexity when run with different).
# I decided to let it be in the definition for now, though, since I didn't make it general and since the lambdas are not dynamic as of now

class LM(object):
    
    """
    Creates a language model object which can be trained and tested.
    The language model has the following attributes:
     - vocab: set of strings
     - lam: float, indicating the constant to add to transition counts to smooth them (default to 1)
     - ngram_size: int, the size of the ngrams
    """
    
    def __init__(self, n, vocab=None, smooth='Laplace', lam=1):
        
        self.vocab = vocab
        self.lam = lam
        self.ngram_size = n
        
    def get_ngram(self, sentence, i):
        
        """
        CHANGE AT OWN RISK.
        
        Takes in a list of string and an index, and returns the history and current 
        token of the appropriate size: the current token is the one at the provided 
        index, while the history consists of the n-1 previous tokens. If the ngram 
        size is 1, only the current token is returned.
        
        Example:
        input sentence: ['bos', 'i', 'am', 'home', 'eos']
        target index: 2
        ngram size: 3
        
        ngram = ['bos', 'i', 'am']  
        #from index 2-(3-1) = 0 to index i (the +1 is just because of how Python slices lists) 
        
        history = ('bos', 'i')
        target = 'am'
        return (('bos', 'i'), 'am')
        """
        
        if self.ngram_size == 1:
            return sentence[i]
        else:
            ngram = sentence[i-(self.ngram_size-1):i+1]
            history = tuple(ngram[:-1])
            target = ngram[-1]
            return (history, target)
                    
    def update_counts(self, corpus):
        
        """
        CHANGE AT OWN RISK.
        
        Creates a transition matrix with counts in the form of a default dict mapping history
        states to current states to the co-occurrence count (unless the ngram size is 1, in which
        case the transition matrix is a simple counter mapping tokens to frequencies. 
        The ngram size of the corpus object has to be the same as the language model ngram size.
        The input corpus (passed by providing the corpus object) is processed by extracting ngrams
        of the chosen size and updating transition counts.
        
        This method creates three attributes for the language model object:
         - counts: dict, described above
         - vocab: set, containing all the tokens in the corpus
         - vocab_size: int, indicating the number of tokens in the vocabulary
        """
        
        if self.ngram_size != corpus.ngram_size:
            raise ValueError("The corpus was pre-processed considering an ngram size of {} while the "
                             "language model was created with an ngram size of {}. \n"
                             "Please choose the same ngram size for pre-processing the corpus and fitting "
                             "the model.".format(corpus.ngram_size, self.ngram_size))
        
        self.counts = defaultdict(dict) if self.ngram_size > 1 else Counter()
        for sentence in corpus.sentences:
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    self.counts[ngram] += 1
                else:
                    # it's faster to try to do something and catch an exception than to use an if statement to check
                    # whether a condition is met beforehand. The if is checked everytime, the exception is only catched
                    # the first time, after that everything runs smoothly
                    try:
                        self.counts[ngram[0]][ngram[1]] += 1
                    except KeyError:
                        self.counts[ngram[0]][ngram[1]] = 1
        
        # first loop through the sentences in the corpus, then loop through each word in a sentence
        self.vocab = {word for sentence in corpus.sentences for word in sentence}
        self.vocab_size = len(self.vocab)
    
    def get_unigram_probability(self, ngram):
        
        """
        CHANGE THIS.
        
        Compute the probability of a given unigram in the estimated language model using
        Laplace smoothing (add k).
        
        I didn't change this. I took parts of this code, but I copied it and let this function untouched.
        """
        
        tot = sum(list(self.counts.values())) + (self.vocab_size*self.lam)
        try:
            ngram_count = self.counts[ngram] + self.lam
        except KeyError:
            ngram_count = self.lam
            print(ngram_count, tot)
        
        prob = ngram_count/tot
        
        return prob
    
    def get_ngram_probability(self, history, target):
        
        """
        CHANGE THIS.
        
        Compute the conditional probability of the target token given the history, using 
        Laplace smoothing (add k).
        
        We used the interpolation method, implementing the formula found in this document that explained the proper method
        (I'm not sure how to properly reference something in code, but this is the link, and the formula is at page 8):
        (The document is called "nlp-programming-en-02-bigramlm.pdf", in case you want to google it, since the link returns a pdf)
        
        https://www.google.ro/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiYsvbl967oAhWS6aQKHW3sD3cQFjAAegQIBRAB&url=http%3A%2F%2Fwww.phontron.com%2Fslides%2Fnlp-programming-en-02-bigramlm.pdf&usg=AOvVaw3MjuGgJ99lCXYeUwbmGpUI
        
        """
        
        lambda_1 = 0.85   # weight for the unigrams
        lambda_2 = 0.85   # weight for the bigrams 
        
        # This is for the bigrams 
        try:
            ngram_tot = np.sum(list(self.counts[history].values()))
            try:
                # I deleted the +self.lam because I'm not using Laplace anymore, so I don't need to add lam
                # I decided to keep the form of the try/except statements though, since I'm not familiar enought with them to change them myself
                transition_count = self.counts[history][target]
            except KeyError:
                # I kept all these excepts and added print statements throughout my debugging process
                # I also decided to have them here just in case some exceptions are raised, and the overall
                # probability will not be zero anyways, because I'm adding together the unigrams and bigrams too
                transition_count = 0
        except KeyError:
            transition_count = 0 
            ngram_tot = 0 
            print("Bigram exeption!!!") # This was here just to check if it entered this branch

        try:
            # I used counts[target] so I'm also getting a dict with the values, instead of only an int
            # It took me a long while to figure out a way to avoif an int+dict error (trying to add together the uni and bi gram probab)
            unigram = sum(list(self.counts[target].values())) 
        except:
            print("Unigram exception!!!") # This was testing if this branch got called
        
        # This was implemented following the mathematical formula found in the previously referenced document
        # Specifically, an interpolated bigram model would need weights for both the bigrams and the unigrams
        # P(w_i|w_i-1) = lambda2 * P_ML(w_i|w_i-1) + (1-lambda2) * P(w_i) + lambda1 * P_ML(w_i) + (1-lambda1) * 1/N
        return lambda_2 * (transition_count/ngram_tot) + (1-lambda_2) * ((lambda_1 * unigram/self.vocab_size)) + (1-lambda_1)/self.vocab_size #unigram_probability
    

    def perplexity(self, test_corpus):
        
        """
        Uses the estimated language model to process a corpus and computes the perplexity 
        of the language model over the corpus.
        
        DON'T TOUCH THIS FUNCTION!!!
        """
        
        probs = []
        for sentence in test_corpus.sentences:
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    probs.append(self.get_unigram_probability(ngram))
                else:
                    probs.append(self.get_ngram_probability(ngram[0], ngram[1]))
        
        entropy = np.log2(probs)
        # this assertion makes sure that you retrieved valid probabilities, whose log must be <= 0
        assert all(entropy <= 0)
        
        avg_entropy = -1 * (sum(entropy) / len(entropy))
        
        return pow(2.0, avg_entropy)


# In[ ]:


# Firstly, we tried to increase the size of n (1-6). We got the following perplexities:

# n=1 => Perplex = 744.4517511377531
# n=2 => Perplex = 325.55512508479444  (the baseline)

# After this the scores become extremely large, because I don't think the corpus is large enough
# I tried them just for check for possible saturation points of the curve

# n=3 => Perplex = 1271.9313925486588
# n=4 => Perplex = 5826.303193446068
# n=5 => Perplex = 12004.074900534266
# n=6 => Perplex = 14942.630558523799


# In[ ]:


# Secondly, we tried to change the lam parameter on the bigram model (since that one had the best perplexity this far):

# lam=0.001 => Perplex = 325.55512508479444  (the baseline)
# lam=0.01 => Perplex = 309.76113373959845  (this score is even better than the baseline! can we beat this?)
# lam=0.1 => Perplex = 415.2798099772183
# lam=1 => Perplex = 813.845075367734


# In[ ]:


# The code snippets for the above steps can be found at the end of the file. 
# After these we decided to try the interpolation method on the bigram model.
# Below is the code with all the parameters already assigned (lambdas are hardcored in the actual function for now)


# In[ ]:


# Bigram model with interpolation. Frequency threshold of 10 is applied.

n = 2
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
bigram_model = LM(n, lam=0.001) # lam doesn't matter here, since I'm not using Laplace anymore
bigram_model.update_counts(train_corpus)

test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=bigram_model.vocab)
bigram_model.perplexity(test_corpus)

# For this, the perplexity score is 297.44508617084915. So we also managed to beat the Laplace with add 0.01. YAY!


# In[ ]:





# In[ ]:





# In[12]:


# From this point lower, the code we used to get the above scores can be noticed. 
# I wouldn't recomment running it, since it takes a huge amount of time!


# In[108]:


# example code to run a unigram model with add 0.001 smoothing. Tokens with a frequency count lower than 10
# are replaced with the UNK string
n = 1
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
unigram_model = LM(n, lam=0.001)
unigram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=unigram_model.vocab)
unigram_model.perplexity(test_corpus)


# In[106]:


# example code to run a bigram model with add 0.001 smoothing. The same frequency threshold is applied.
n = 2
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
bigram_model = LM(n, lam=0.001)
bigram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=bigram_model.vocab)
bigram_model.perplexity(test_corpus)


# In[ ]:


# The output perplexity is 325.55512508479444
# This is the baseline score that I have to beat (so I have to have a perplexity that's lower than 325)


# In[17]:


# example code to run a trigram model with add 0.001 smoothing. The same frequency threshold is applied.
n = 3
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
trigram_model = LM(n, lam=0.001)
trigram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=trigram_model.vocab)
trigram_model.perplexity(test_corpus)


# In[19]:


# example code to run a fourgram model with add 0.001 smoothing. The same frequency threshold is applied.
n = 4
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
fourgram_model = LM(n, lam=0.001)
fourgram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=fourgram_model.vocab)
fourgram_model.perplexity(test_corpus)


# In[20]:


# The output perplexity is 5826.303193446068


# In[21]:


# example code to run a fivegram model with add 0.001 smoothing. The same frequency threshold is applied.
n = 5
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
fivegram_model = LM(n, lam=0.001)
fivegram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=fivegram_model.vocab)
fivegram_model.perplexity(test_corpus)


# In[22]:


# The output perplexity is 12004.074900534266


# In[23]:


# example code to run a sixgram model with add 0.001 smoothing. The same frequency threshold is applied.
n = 6
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
sixgram_model = LM(n, lam=0.001)
sixgram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=sixgram_model.vocab)
sixgram_model.perplexity(test_corpus)


# In[24]:


# The output perplexity is 14942.630558523799


# In[25]:


# Change the lam=0.001 of the model to see if the perplexity lowers for a bigram


# In[107]:


# example code to run a bigram model with add 0.01 smoothing. The same frequency threshold is applied.
n = 2
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
bigram_model_01 = LM(n, lam=0.01)
bigram_model_01.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=bigram_model_01.vocab)
bigram_model_01.perplexity(test_corpus)


# In[27]:


# The output perplexity is 309.76113373959845
# This is the best perplexity this far!


# In[34]:


# example code to run a bigram model with add 0.1 smoothing. The same frequency threshold is applied.
n = 2
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
bigram_model_1 = LM(n, lam=0.1)
bigram_model_1.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=bigram_model_1.vocab)
bigram_model_1.perplexity(test_corpus)


# In[29]:


# The output perplexity is 415.2798099772183


# In[30]:


# example code to run a bigram model with add 1.0 smoothing. The same frequency threshold is applied.
n = 2
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
bigram_model_11 = LM(n, lam=1.0)
bigram_model_11.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=bigram_model_11.vocab)
bigram_model_11.perplexity(test_corpus)


# In[31]:


# The output perplexity is 813.845075367734


# In[ ]:




