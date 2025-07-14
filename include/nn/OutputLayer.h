#pragma once

#include "Layer.h"
#include "../Data.h"
class OutputLayer : public Layer
{
private:
    /* data */
public:
    OutputLayer(int num_neurons, int previous_layer_size)
        : Layer(num_neurons, previous_layer_size) {}
    void feedForward(Layer& prev);
    void backProp(Data<double>& data);
    void updateWeights(double learning_rate, Layer& layer);
};


