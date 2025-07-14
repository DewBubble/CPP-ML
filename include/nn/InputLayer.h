#pragma once
#include "Layer.h"
#include "../Data.h"
class InputLayer: public Layer {

public:
    InputLayer( int num_neurons, int previous_layer_size)
        : Layer(num_neurons, previous_layer_size) {}
    void setLayerOutputs(Data<double>& d);

};