from string import punctuation, digits
import numpy as np
import random




def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    p = label*(np.inner(feature_vector,theta)+theta_0)
    if p>=1:
        return 0
    else:
        return 1-p



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    n = feature_matrix.shape[0]
    s = 0
    for i in range(n):
        s += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
    return s/n




def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    theta = current_theta 
    theta_0 = current_theta_0
    if label*(np.inner(current_theta,feature_vector)+current_theta_0)<=0.0001:
            theta = theta + label*feature_vector
            theta_0 = theta_0 + label
    return [theta,theta_0]



def perceptron(feature_matrix, labels, T):
    n = feature_matrix.shape[0]
    m = feature_matrix.shape[1]
    theta_0 = 0
    theta = np.zeros((m,))
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            result = perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
            theta = result[0]
            theta_0 = result[1]
    # Your code here
    return [theta,theta_0]



def average_perceptron(feature_matrix, labels, T):
    n = feature_matrix.shape[0]
    m = feature_matrix.shape[1]
    theta_0 = 0
    theta = np.zeros((m,))
    sum_t = theta
    sum_t0 = 0
    counter = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            result = perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
            theta = result[0]
            theta_0 = result[1]
            sum_t += theta
            sum_t0 += theta_0
            counter += 1
    # Your code here
    sum_t = sum_t*(1/counter)
    sum_t0 = sum_t0/counter
    return [sum_t,sum_t0]


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    if label*(np.inner(theta,feature_vector)+theta_0)<=1:
        theta = ( 1 - eta*L)*theta + eta*label*feature_vector
        theta_0 = theta_0 + eta*label
    else:
        theta = ( 1 - eta*L)*theta
    return [theta,theta_0]



def pegasos(feature_matrix, labels, T, L):
    n = feature_matrix.shape[0]
    m = feature_matrix.shape[1]
    theta_0 = 0
    theta = np.zeros((m,))
    t = 1
    eta = 1
    for i in range(T):
        for l in get_order(feature_matrix.shape[0]):
            result = pegasos_single_step_update(feature_matrix[l],labels[l],L,eta,theta,theta_0)
            theta = result[0]
            theta_0 = result[1]
            t += 1
            eta = 1/np.sqrt(t)
    return [theta,theta_0]






def classify(feature_matrix, theta, theta_0):
    n = feature_matrix.shape[0]
    classification = np.zeros((n))
    for i in range(n):
        x = feature_matrix[i]
        if (np.inner(x,theta) + theta_0)>0.0001:
            classification[i]=1
        else:
            classification[i]=-1
    return classification


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    result = classifier(train_feature_matrix,train_labels,**kwargs)
    theta = result[0]
    theta_0 = result[1]
    train_classification = classify(train_feature_matrix,theta,theta_0)
    train_accuracy = accuracy(train_classification,train_labels)

    result2 = classifier(val_feature_matrix,val_labels,**kwargs)
    theta2 = result2[0]
    theta_02 = result2[1]
    val_classification = classify(val_feature_matrix,theta2,theta_02)
    val_accuracy = accuracy(val_classification,val_labels)
    return [train_accuracy,val_accuracy]



def extract_words(text):
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=True):    
    indices_by_word = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        if remove_stopword == False:
            stopword = []
        else:
            with open('stopwords.txt','r') as f:
                content = f.read()
                stopword = content.split("\n")
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    # Your code here
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        skip = True
    return feature_matrix



def accuracy(preds, targets):
    return (preds == targets).mean()
