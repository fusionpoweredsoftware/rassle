class LossFunction {
public:
    LossFunction(float discount_factor) : discount_factor_(discount_factor) {}

    float compute_loss(std::vector<Transition> transitions) {
        float loss = 0.0f;
        for (Transition t : transitions) {
            float target = t.reward;
            if (!t.done) {
                // Bellman equation for Q-learning
                target += discount_factor_ * max(model_.predict(t.next_state));
            }
            loss += (target - model_.predict(t.state, t.action))^2;
        }
        return loss;
    }

private:
    float discount_factor_;
    Model model_;
};
