#include "../include/nn/Neuron.h"
Neuron::Neuron( int previousLayerSize, int currentLayerSize) {
    
    initializeWeights( previousLayerSize);
}

void Neuron::initializeWeights(int previousLayerSize) {
    for (int i = 0; i < previousLayerSize+1; ++i) {
        weights.push_back(gen());
    }
}

const std::vector<double>& Neuron::getWeights() const {
    return weights;
}

void Neuron::updateWeights(const std::vector<double>& inputs, double learning_rate) {
    for (size_t i = 0; i < weights.size() - 1; ++i) {
        weights[i] += learning_rate * delta * inputs[i];
    }
    weights.back() += learning_rate * delta; // Update bias
}