import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # repo root
CSV = ROOT / "data" / "raw" / "iris.csv"

# Function to plot data
def visualize_data(input_ax, x_0, y_0, x_1, y_1):
    
    # Plot title
    input_ax.set_title("Perceptron Learning Algorithm")

    # Axis titles
    input_ax.set_xlabel("Sepal Length")
    input_ax.set_ylabel("Sepal Width")

    input_ax.scatter(x_0, y_0, marker='x', edgecolors='red')
    input_ax.scatter(x_1, y_1, marker='x', edgecolors='blue')

    # Locking the plot scale
    input_ax.set_xlim(input_ax.get_xlim())
    input_ax.set_ylim(input_ax.get_ylim())

# Function to read the csv dataset
def read_dataset():

    return(pd.read_csv(CSV))

# Function to create sample/label pairs
def training_data(input: pd.DataFrame):

    # Refining the dataframe to only have examples of 'Setosa' and 'Virginica' labels
    refined_data = (input.loc[input['variety'].isin(['Setosa', 'Virginica'])].copy())

    # Mapping Setosa and Virginica labels to 0 and 1 respectively
    refined_data['variety'] = refined_data['variety'].map({'Setosa': 0, 'Virginica': 1})
    
    # For Class 0:
    # Features
    sepal_length_0 = refined_data[refined_data['variety'] == 0]["sepal.length"]
    sepal_width_0 = refined_data[refined_data['variety'] == 0]["sepal.width"]
    
    # Labels
    labels_0 = refined_data[refined_data['variety'] == 0]['variety']


    # For Class 1:
    # Features
    sepal_length_1 = refined_data[refined_data['variety'] == 1]["sepal.length"]
    sepal_width_1 = refined_data[refined_data['variety'] == 1]["sepal.width"]

    labels_1 = refined_data[refined_data['variety'] == 1]['variety']

    return(sepal_length_0.to_numpy(), sepal_width_0.to_numpy(), sepal_length_1.to_numpy(), sepal_width_1.to_numpy(), labels_0.to_numpy(), labels_1.to_numpy())


# Class to implement perceptron learning algorithm
class Perceptron:

    def __init__(self, learning_rate, iterations):

        # Pyplot
        self.saved_plot = None
        self.last_line = None
        self.line_of_best_fit = None
        self.counter_text = None
        self.weights_text = None
        # --------------------------


        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation = self.activation_function
        self.errors_per_epoch = []
        self.weights = None
        self.bias = None

    # Perceptron learning algorithm
    def learning(self, X, y, input_plt):

        # Initializing weights (zeros)
        self.weights = np.zeros(X.shape[1])

        self.saved_plot = input_plt
        self.counter_text = self.saved_plot.text(
            0.02, 0.95, "", transform=self.saved_plot.transAxes, fontsize=12, color="black"
        )
        
        self.weights_text = self.saved_plot.text(
            0.63, 0.95, "", transform=self.saved_plot.transAxes, fontsize=12, color="black"
        )

        # Initializing bias
        self.bias = 0

        self.weights_text.set_text(f"Weights: {self.weights}")

        # For a specified number of epochs
        for _ in range(self.iterations):

            self.counter_text.set_text(f"Epoch: {_ + 1}")
            
            # (misclassification counter variable)
            errors = 0

            # Iterate through each sample in training dataset
            for i in range(0, len(X)):

                # For each sample, compute w^T*x_i + b
                h_x = np.dot(self.weights, X[i]) + self.bias

                # Apply the activation function
                model_prediction = self.activation(h_x)

                # If the model's prediction does not match the sample's ground truth label
                if (model_prediction != y[i]):

                    # Update weights: w <-- w + 𝛼*x_i*y_i
                    self.weights = self.weights + X[i] * (self.learning_rate * (y[i] - model_prediction))

                    # Update bias: b <-- b + 𝛼*y_i
                    self.bias = self.bias + self.learning_rate * (y[i] - model_prediction)

                    # (increment misclassification counter)
                    errors += 1

                    # Compute decision boundary line
                    if (_ % 10 == 0):
                        
                        self.decision_boundary_line(final=False)

                    self.weights_text.set_text(f"Weights: {self.weights}")

            if (errors > 0):

                self.errors_per_epoch.append(errors)

            # After iterating through all samples, if there are no errors, end the algorithm
            if (errors == 0):

                self.decision_boundary_line(final=True)
                self.counter_text.set_text(f"Epoch: {_ + 1}")
                self.weights_text.set_text(f"Weights: {self.weights}")
                return(self.weights, self.bias)

    # Decision Boundary Line
    def decision_boundary_line(self, final):

        # Clearing the last line from the plot (if there is one)
        if self.last_line:

            self.last_line.remove()


        # Given that the weight vector is w ∈ ℝ^d, we compute the decision boundary line using
        # w_1*x + w_2*y + b = 0
        # y = -(w_1 / w_2)x - (b / w_2)
        x = np.linspace(*self.saved_plot.get_xlim(), 100)
        y = -(self.weights[0] / self.weights[1]) * x - (self.bias / self.weights[1])

        line_color = "red" if final else "green"
        self.last_line, = self.saved_plot.plot(x, y, color=line_color)
        
        plt.pause(0.3)


    # Activation Function (Step Function)
    def activation_function(self, model_output):
        
        # Step function -> f(z) = 1 if f(z) ≥ 0 else 0
        return 1 if model_output >= 0 else 0

# Main function
if __name__ == "__main__":

    # Open the training dataset
    dataset = read_dataset()

    # Retrieving features and labels from dataset: features - "sepal.length", "sepal.width"
    sepal_length_0, sepal_width_0, sepal_length_1, sepal_width_1, labels_0, labels_1 = training_data(dataset)
    
    # Creating Perceptron model with (learning rate) 𝛼 = 0.01 and (epochs) t = 1000
    model = Perceptron(0.1, 1000)

    # Stacking features per class
    features_0 = np.column_stack((sepal_length_0, sepal_width_0))
    features_1 = np.column_stack((sepal_length_1, sepal_width_1))

    # Stacking features together
    features = np.vstack((features_0, features_1))
    labels = np.concatenate((labels_0, labels_1))

    # Visualizing training dataset
    fig, ax = plt.subplots()
    visualize_data(ax, sepal_length_0, sepal_width_0, sepal_length_1, sepal_width_1)

    # Feeding the training dataset to perceptron model
    print(f"Final weights and bias: {model.learning(features, labels, ax)}")

    plt.show()

    # Plotting convergence (errors)
    plt.figure()
    plt.title("Perceptron Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Misclassifications")
    plt.plot(model.errors_per_epoch, marker="o")

    plt.show()

