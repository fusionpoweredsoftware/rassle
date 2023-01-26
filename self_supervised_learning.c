void self_supervised_learning() {
    // Load preprocessed data
    vector<string> data = load_preprocessed_data();

    // Define network architecture
    NeuralNetwork net;
    net.add_layer(new FullyConnectedLayer(500, data[0].size()));
    net.add_layer(new ReLUActivation());
    net.add_layer(new FullyConnectedLayer(250, 500));
    net.add_layer(new ReLUActivation());
    net.add_layer(new FullyConnectedLayer(125, 250));
    net.add_layer(new ReLUActivation());
    net.add_layer(new FullyConnectedLayer(64, 125));

    // Define loss function
    LossFunction *loss = new MeanSquaredError();

    // Define optimizer
    Optimizer *optimizer = new Adam(net, loss);

    // Train network
    for (int i = 0; i < data.size(); i++) {
        vector<double> input = convert_to_input_vector(data[i]);
        vector<double> output = convert_to_output_vector(data[i]);
        optimizer->train(input, output);
    }

    // Save trained network
    save_trained_network(net);
}