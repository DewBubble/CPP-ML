#pragma once
#include <vector>
#include "../Utility.h"
class Neuron {
    std::vector<double> weights;
    double output=0;
    double delta=0;

    std::function<double()> gen = createRandomNumberGenerator(-1.0, 1.0);

    public:
    Neuron( int previousLayerSize, int currentLayerSize);
    void initializeWeights(int previousLayerSize);
    double getDelta() const { return delta; }
    void setDelta(double d) { delta = d; }
    double getOutput() const { return output; }
    void setOutput(double value) { output = value; }
    const std::vector<double>& getWeights() const;

    void updateWeights(const std::vector<double>& inputs, double learning_rate);
};