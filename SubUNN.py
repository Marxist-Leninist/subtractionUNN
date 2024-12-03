import random
import time

# Unary Neural Network class
class UnaryNN:
    def __init__(self, max_value):
        self.max_value = max_value
        # Initialize weights to 1 for each possible input value up to max_value
        self.weights = [1 for _ in range(self.max_value + 1)]

    def forward(self, input1, input2):
        # Compute output for subtraction: input1 - input2
        # Both inputs are counts in unary (non-negative integers)
        hidden_count = 0

        # Sum weights up to input1
        for i in range(min(input1, self.max_value)):
            hidden_count += self.weights[i]

        # Subtract weights up to input2
        for i in range(min(input2, self.max_value)):
            hidden_count -= self.weights[i]

        # Adjust for inputs beyond max_value
        if input1 > self.max_value:
            hidden_count += (input1 - self.max_value) * self.weights[-1]
        if input2 > self.max_value:
            hidden_count -= (input2 - self.max_value) * self.weights[-1]

        # Ensure output is non-negative
        output = max(0, hidden_count)
        return output

    def train(self, input1, input2, target_output):
        predicted_output = self.forward(input1, input2)
        error = target_output - predicted_output

        if error != 0:
            error_sign = 1 if error > 0 else -1
            # Adjust weights up to the maximum of input1 and input2
            max_input = min(max(input1, input2), self.max_value)
            for i in range(max_input):
                # Adjust weights for input1
                if i < input1:
                    self.weights[i] += error_sign
                    # Ensure weights stay at least 1
                    self.weights[i] = max(1, self.weights[i])
                # Adjust weights for input2
                if i < input2:
                    self.weights[i] -= error_sign
                    self.weights[i] = max(1, self.weights[i])
            # Optionally, cap the weights at a maximum value to prevent them from growing too large
            for i in range(max_input):
                self.weights[i] = min(100, self.weights[i])  # Cap at 100

    def train_model(self, epochs):
        print('Starting training...')
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            # Generate random inputs within the range
            input1 = random.randint(0, self.max_value)
            input2 = random.randint(0, self.max_value)
            target_output = max(0, input1 - input2)
            # Train the network on this sample
            self.train(input1, input2, target_output)

            if epoch % 10000 == 0 or epoch == 1:
                print('Completed {} epochs'.format(epoch))
        end_time = time.time()
        print('Training completed in {:.2f} seconds.'.format(end_time - start_time))

    def test(self, num_tests):
        print('\nRunning {} automated tests...'.format(num_tests))
        start_time = time.time()
        correct = 0
        for _ in range(num_tests):
            input1 = random.randint(0, self.max_value * 2)  # Test beyond training range
            input2 = random.randint(0, self.max_value * 2)
            expected_output = max(0, input1 - input2)
            predicted_output = self.forward(input1, input2)
            if predicted_output == expected_output:
                correct += 1
        accuracy = (correct * 100) / num_tests
        end_time = time.time()
        print('Accuracy: {:.2f}%'.format(accuracy))
        print('Testing completed in {:.2f} seconds.'.format(end_time - start_time))

    def interactive_test(self):
        while True:
            try:
                a = int(input('\nEnter minuend (negative number to exit): '))
                if a < 0:
                    break
                b = int(input('Enter subtrahend: '))
                start_time = time.time()
                predicted_output = self.forward(a, b)
                end_time = time.time()
                actual_output = max(0, a - b)
                print('Neural network output: {}'.format(predicted_output))
                print('Actual result: {}'.format(actual_output))
                print('Computation time: {:.9f} seconds'.format(end_time - start_time))
            except ValueError:
                print('Please enter valid integers.')

def main():
    max_value = 100    # Increased to handle larger inputs
    epochs = 100000    # Number of training epochs
    num_tests = 10000  # Increased number of tests for evaluation

    # Initialize the unary neural network
    nn = UnaryNN(max_value)

    # Train the network
    nn.train_model(epochs)

    # Test the network
    nn.test(num_tests)

    # Interactive testing
    nn.interactive_test()

if __name__ == '__main__':
    main()
