class Optimizer {
private:
    double learning_rate;
    double momentum;
    double weight_decay;
    vector<double> gradient;
    vector<double> previous_update;

public:
    Optimizer(double learning_rate, double momentum, double weight_decay) {
        this->learning_rate = learning_rate;
        this->momentum = momentum;
        this->weight_decay = weight_decay;
    }

    void update(vector<double>& weights, vector<double>& gradient) {
        if (previous_update.size() == 0) {
            previous_update.resize(weights.size());
        }

        // Update gradient
        for (int i = 0; i < weights.size(); i++) {
            previous_update[i] = momentum * previous_update[i] - learning_rate * gradient[i];
            weights[i] += previous_update[i];
        }
    }
};
