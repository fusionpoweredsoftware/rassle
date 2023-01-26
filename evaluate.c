void evaluate() {
    // Load preprocessed data and trained agent
    vector<string> data = load_preprocessed_data();
    Agent agent = load_trained_agent();

    // Initialize evaluation metrics
    double total_reward = 0;
    int num_correct = 0;

    // Evaluate agent on data
    for (int i = 0; i < data.size(); i++) {
        // Get current state and action
        State current_state = get_current_state(data[i]);
        Action action = agent.get_action(current_state);

        // Take action and observe new state and reward
        State new_state = take_action(current_state, action);
        double reward = observe_reward(new_state);

        // Update evaluation metrics
        total_reward += reward;
        if (action == correct_action(data[i])) {
            num_correct++;
        }
    }

    // Print evaluation results
    double accuracy = (double) num_correct / data.size();
    double avg_reward = total_reward / data.size();
    cout << "Accuracy: " << accuracy << endl;
    cout << "Average reward: " << avg_reward << endl;
}