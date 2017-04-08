# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import math
import numpy as np
import scipy

class MaxEnt(Classifier):
    def __init__(self, model=[]):
        super(Classifier, self).__init__()
    def get_model(self): return None
    def set_model(self, model): pass
    model = property(get_model, set_model)
    fmatrix = None #to hold lambdas
    labels = None #list of labels in data
    features = None #list of features being used
    labeldict = None #convert between labels and ints
    featdict = None #convert between features and ints
    accuracy = [0.0]
    
    
    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        #create a list of all non-stopword in instances
        vocab = set()
        for instance in instances:
            for word in instance.features():
                vocab.add(word)
        vocab = list(vocab)
    
        #get list of labels in data
        labels= set()
        for instance in instances:
            if instance.label not in labels:
                labels.add(instance.label)
        labels = list(labels)
        self.labels = labels
        self.features = vocab  
        
        #create feature matrix
        featurematrix = np.zeros((len(self.labels),len(self.features)), dtype = float)
        self.fmatrix = featurematrix
        
        #map each label and feature to ints
        featdict = {}
        for i in range(len(self.features)):
            featdict[self.features[i]] = i
            featdict[i] = self.features[i]
        labeldict = {}
        for i in range(len(self.labels)):
            labeldict[self.labels[i]] = i
            labeldict[i] = self.labels[i]   
        self.featdict = featdict
        self.labeldict = labeldict
        
        self.train_sgd(instances, dev_instances, 0.0001, 30)
        
    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient"""
        #get minibatch
        start = 0
        stop = batch_size
        batch = train_instances[start:stop]
        self.learn_batch(batch, learning_rate) #initial learning
        
        flag = 1 #to count number of training iterations
        done = False #to indicate convergence
        exp1 = [0] #to count number of times gradient's been calculated
        
        while done == False:
            start = stop
            stop = stop + batch_size
            if stop == len(train_instances):
                start = 0
                stop = batch_size
            self.learn_batch(train_instances[start:stop], learning_rate)
            
            #after so many training iterations, test on dev
            if flag == 10:
                exp1.append(flag + exp1[len(exp1)-1])
                self.accuracy.append(self.testOnDev(dev_instances))
                done = self.test_accuracy()
                flag = 1
                #shuffle instances after each dev set test
                np.random.shuffle(train_instances)
            flag += 1
        print 'Accuracy:', self.accuracy
        print 'DataPoints:', exp1
    
    #update parameters based on a single mini-batch
    def learn_batch(self, batch, learning_rate):
        #create observed and expected count matrices and probability matrix
        observed = np.zeros((len(self.labels), len(self.features)))
        expected= np.zeros((len(self.labels), len(self.features)))
        gradient = np.zeros((len(self.labels), len(self.features))) 
        
        #get observed counts of features per label in minibatch
        for instance in batch:
            for feature in instance.features():
                    observed[self.labeldict[instance.label]][self.featdict[feature]] += 1
        
        #get posterior for each instance, update expected values
        for instance in batch:
            denom = self.get_denom(instance.features())
            for i in range(len(self.labels)):
                num = self.get_num(instance.features(), i)
                for word in instance.features():
                    expected[i][self.featdict[word]] += math.exp(num-denom)
            
        #calculate gradient, update lambdas
        for i in range(len(self.labels)):
            for j in range(len(self.features)):
                gradient[i][j] = observed[i][j] - expected[i][j]
                self.fmatrix[i][j] += gradient[i][j] * learning_rate   
                  
    #calculate numerator for posterior
    def get_num(self, features, i):
        value = 0
        for feature in features:
            if feature in self.features:
                value += self.fmatrix[i][self.featdict[feature]]
        return value
            
    #calculate denominator for posterior
    def get_denom(self, features):
        values = []
        value = 0
        for i in range(len(self.labels)):
            for feature in features:
                if feature in self.features:
                    value += (self.fmatrix[i][self.featdict[feature]])
            values.append(value)
            value = 0
        return scipy.misc.logsumexp(values)
  
    #classify dev instances, get accuracy      
    def testOnDev(self, dev):
        correct = [self.classify(x) == x.label for x in dev]
        return (100.0 * float(sum(correct)) / float(len(correct)))
        
   #return true if accuracy has converged
    def test_accuracy(self):
        l = len(self.accuracy)
        if l >= 5:
            #if accuracy hasn't changed after 3 dev set runs
            if (self.accuracy[l-1] == self.accuracy[l-2]):
                if (self.accuracy[l-2] == self.accuracy[l-3]):
                    if (self.accuracy[l-3] == self.accuracy[l-4]):
                        if (self.accuracy[l-4] == self.accuracy[l-5]):
                            return True
        return False
                
    def classify(self, instance):    
        probs = np.zeros(len(self.labels)) #vector to hold logprobs for labels
        for i in range(len(self.labels)):
            probs[i] -= (self.get_num(instance.features(),i))
        return self.labeldict[np.argmin(probs)] 

