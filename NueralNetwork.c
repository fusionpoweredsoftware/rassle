class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int output_size, vector<int> hidden_sizes) {
        // Create layers
        this->layers.push_back(new Layer(input_size, hidden_sizes[0]));
        for (int i = 1; i < hidden_sizes.size(); i++) {
            this->layers.push_back(new Layer(hidden_sizes[i-1], hidden_sizes[i]));
        }
        this->layers.push_back(new Layer(hidden_sizes[hidden_sizes.size()-1], output_size));

        // Initialize weights and biases
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->initialize_weights();
            layers[i]->initialize_biases();
        }
    }

    vector<double> forward(vector<double> input) {
        // Pass input through layers
        vector<double> current_output = input;
        for (int i = 0; i < layers.size(); i++) {
            current_output = layers[i]->forward(current_output);
        }
        return current_output;
    }

    void backward(vector<double> input, vector<double> output, vector<double> target) {
        // Compute error for output layer
        vector<double> error = vector_sub(target, output);

        // Backpropagate error through layers
        for (int i = layers.size()-1; i >= 0; i--) {
            error = layers[i]->backward(input, output, error);
            input = layers[i]->get_output();
        }
    }

    void update_weights(double learning_rate) {
        // Update weights for all layers
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->update_weights(learning_rate);
        }
    }

    void update_biases(double learning_rate) {
        // Update biases for all layers
        for (int i = 0; i < layers.size(); i++) {
            layers[i]->update_biases(learning_rate);
        }
    }

private:
    vector<Layer*> layers;
};