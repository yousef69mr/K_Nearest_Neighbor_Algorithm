from math import sqrt
import numpy as np
import pandas as pd


def normalize_data(data):
    outputs = data.iloc[:, -1]
    features = data.iloc[:, :-1]

    # print(outputs)
    # print(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean)/std
    # print("Mean : ",list(mean),"\nStd:",list(std))

    normalized_data = features.loc[:, outputs.name] = outputs
    # print("\n\n", features)
    return normalized_data, list(mean), list(std)


def normalize_row(row, mean, std):
    result = list()
    for i in range(len(row)-1):
        result.append((row[i] - mean[i])/std[i])
    result.append(row[-1])
    return result


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
        # sort according to second element
        distances.sort(key=lambda tup: tup[1])
    # print(distances)
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def split_data(dataset, percent):
    training_data = dataset.sample(frac=percent, random_state=25)
    testing_data = dataset.drop(training_data.index)
    return training_data, testing_data


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    # print(set(output_values))

    # key number of instances in output_values
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def calculate_accuracy(predicted_values, actual_values):
    errors = list()
    for i in range(len(actual_values)):
        if predicted_values[i] != actual_values[i]:
            errors.append(predicted_values)

    print('Number of correctly classified instances : %d' % (len(actual_values)-len(errors)))
    print('Total number of instances : %d' % len(actual_values))
    accuracy = ((len(actual_values)-len(errors))/len(actual_values))*100
    return accuracy


def run():
    dataset = pd.read_csv("BankNote_Authentication.csv")
    # print(dataset)
    training, testing = split_data(dataset, 0.7)
    print(training)
    normalized_training, mean, std = normalize_data(training)

    training = np.array(training)
    testing = np.array(testing)
    # while k_factor is not int:
    # k_factor = int(input("Enter K Factor : "))
    k_factors = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for k in k_factors:
        predictions = list()
        outputs = list()

        # predictions
        for i in range(len(testing)):
            normalized_test = normalize_row(testing[i], mean, std)
            outputs.append(normalized_test[-1])
            prediction = predict_classification(training, normalized_test, k)
            predictions.append(prediction)
            # print('Expected %d, Got %d' % (normalized_test[-1], prediction))

        # print(predictions, len(predictions))
        print("/************************************************/")
        print("K Factor : %d " % k)
        accuracy = calculate_accuracy(predictions, outputs)
        print('Accuracy : %d ' % accuracy, '%')
        print("/************************************************/\n")


if __name__ == '__main__':
    run()


