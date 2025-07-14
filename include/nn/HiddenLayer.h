#pragma once
#include "Layer.h"
class HiddenLayer : public Layer {
public:
    HiddenLayer(int num_neurons, int previous_layer_size)
        : Layer(num_neurons, previous_layer_size) {}

    void feedForward(Layer prev);
    void backProp(Layer next);
    void updateWeights(double learning_rate, Layer& layer);
};