class RewardFunction {
public:
    virtual double operator()(const State& state, const Action& action) = 0;
};

class PredictionAccuracyReward : public RewardFunction {
public:
    double operator()(const State& state, const Action& action) {
        double reward = 0;
        // Compare the predicted output with the actual output
        if (state.predicted_output == state.actual_output) {
            reward = 1;
        } else {
            reward = -1;
        }
        return reward;
    }
};
