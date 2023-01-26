class State {
public:
    vector<double> features;
    int label;
    State(vector<double> f, int l) : features(f), label(l) {}
};