#pragma once

#include "Neuron.h"
#include <vector>

class Layer {
public:
    Layer(int num_neurons, int num_inputs_per_neuron);
    
    int getSize(){
        return neurons.size();
    }

    std::vector<Neuron>& get_neurons();
private:

    std::vector<Neuron> neurons;
    std::vector<double> output;
};