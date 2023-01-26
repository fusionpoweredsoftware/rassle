void data_preprocessing(string data) {
    // Tokenization
    vector<string> tokens = tokenize(data);

    // Stemming
    for (int i = 0; i < tokens.size(); i++) {
        tokens[i] = stem(tokens[i]);
    }

    // Stop word removal
    vector<string> filtered_tokens;
    for (int i = 0; i < tokens.size(); i++) {
        if (!is_stop_word(tokens[i])) {
            filtered_tokens.push_back(tokens[i]);
        }
    }

    // Save preprocessed data
    save_preprocessed_data(filtered_tokens);
}