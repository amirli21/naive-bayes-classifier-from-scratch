from typing import Iterable, List, Sequence, Any
from math import sqrt, exp, pi
from csv import reader
from random import seed
from random import randrange

Dataset = Iterable


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def separate_by_class(dataset: Dataset) -> dict:
    """
    Splits the given dataset by class values.
    @param dataset: labeled dataset in iterable format
    @return: a dictionary where each key is the class
             value and values are list of all the records belonging to the class
    """
    separated = dict()
    for row in range(len(dataset)):
        vector = dataset[row]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


def mean(data: Sequence) -> float:
    """
    Calculates mean of the given 1D data.
    @param data: Sequence of numbers.
    @return: a float which is the
              population mean of the given data.
    """
    return sum(data) / float(len(data))  # float len data just for avoiding the ZeroDivisionError


def sample_stddev(data: Sequence) -> float:
    """
    Calculates the sample standard deviation
                  of 1D data with 1 degree of
                   freedom.
    @param data: Sequence of numbers.
    @return: a float which is the sample
             standard deviation of the given
             data.
    """
    average = mean(data)
    variance = sum([(x - average) ** 2 for x in data]) / float(len(data) - 1)
    return sqrt(variance)


def summarize_dataset(dataset: Dataset) -> List:
    """
    Calculates the mean, sample_stddev and count for each column in a dataset
    @param dataset: Sequence of numbers.
    @return: Summary of the dataset.
    """
    summaries = [(mean(column), sample_stddev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


def summarize_by_class(dataset: Dataset) -> dict[Any, list]:
    """
    Calculates the mean, sample_stddev and
     count for each class in the dataset.
     @param dataset: Sequence of numbers.
     @return: Summary of the dataset for each class.
    """
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


def calculate_probability(x: float, mean_: float, stddev: float):
    """
    Calculates the Gaussian probability distribution for x.For more information,
    please refer to https://en.wikipedia.org/wiki/Gaussian_function.
    @param x: Value to calculate the probability.
    @param mean_: Mean of the dataset.
    @param stddev: Sample standard deviation of the dataset.
    @return: Probability of the given value.
    """
    exponent = exp(-((x - mean_) ** 2 / (2 * stddev ** 2)))
    return (1 / (sqrt(2 * pi) * stddev)) * exponent


def calculate_class_probabilities(summaries: dict, row: List):
    """

    @param summaries: Class summaries of the given dataset.
    @param row: Row of the dataset
    @return: Probability values for each class.
    """
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean_, stddev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean_, stddev)
    return probabilities


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return predictions


# Test Naive Bayes on Iris Dataset
seed(1)
filename = r'../datasets/iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
