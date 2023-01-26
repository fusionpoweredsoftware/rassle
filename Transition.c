class Transition {
public:
    Transition(State state, Action action, float reward, State next_state, bool done)
        : state_(state), action_(action), reward_(reward), next_state_(next_state), done_(done) {}

    State state_;
    Action action_;
    float reward_;
    State next_state_;
    bool done_;
};
