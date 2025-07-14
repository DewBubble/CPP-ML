#include "../include/nn/Layer.h"

Layer::Layer(int previousLayerSize, int currentLayerSize) {
    for (int i = 0; i < currentLayerSize; ++i) {
        neurons.push_back(Neuron(previousLayerSize, currentLayerSize));
    }
}

 std::vector<Neuron>& Layer::get_neurons() {
    return neurons;
}