from corpus import Document, NamesCorpus, ReviewCorpus
#from key_maxent import MaxEnt
from maxent import MaxEnt
from unittest import TestCase, main
from random import shuffle, seed
import sys
from nltk.corpus import stopwords


class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        words = []
        for word in self.data.split():
            if word not in stopwords.words('english'):
                words.append(word)
        return words
        
class SentimentWords(Document):
    def features(self):
        vocab=['bad','terrible', 'great','good','okay']
        words = []
        for word in self.data.split():
            if word in vocab: 
                words.append(word)
        return words
        

class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]] 

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return float(sum(correct)) / len(correct)

class MaxEntTest(TestCase):
    u"""Tests for the MaxEnt classifier."""

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training, dev, and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:5000], names[5000:6000], names[6000:])

#    def test_names_nltk(self):
#        """Classify names using NLTK features"""
#        train, dev, test = self.split_names_corpus()
#        classifier = MaxEnt()
#        classifier.train(train, dev)
#        acc = accuracy(classifier, test)
#        self.assertGreater(acc, 0.70)

    def split_review_corpus(self, document_class):
        """Split the yelp review corpus into training, dev, and test sets"""
        reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
        seed(hash("reviews"))
        shuffle(reviews)
        #return (reviews[:1000], reviews[1000:1100], reviews[1100:1400])
        return (reviews[:100000], reviews[10000:11000], reviews[11000:14000])

#    def test_reviews_bag(self):
#        """Classify sentiment using bag-of-words"""
#        train, dev, test = self.split_review_corpus(BagOfWords)
#        classifier = MaxEnt()
#        classifier.train(train, dev)
#        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_reviews_words(self):
        """Classify sentiment using sentiment words"""
        train, dev, test = self.split_review_corpus(SentimentWords)
        classifier = MaxEnt()
        classifier.train(train, dev)
        self.assertGreater(accuracy(classifier, test), 0.55)


if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)

