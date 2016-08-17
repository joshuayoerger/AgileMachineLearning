from sknn.mlp import Layer, Classifier


def wrapper_for_backprop_neural_network_code(train_x, train_y, test_x, test_y):
    # Class Mob Solution
    score = None

    clf = Classifier(
        layers=[Layer('Sigmoid', units=10), Layer('Softmax')],
        learning_rate=.001, n_iter=1
    )

    for i in range(100):
        clf.partial_fit(train_x, train_y)
        score = clf.score(test_x, test_y)
        print(i, score)
    #nn.fit(train_x, train_y)

    #predicted = nn.predict(test_x)
    #score = accuracy_score(predicted, test_y)
    print(score)
    return score
