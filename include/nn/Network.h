#pragma once

#include "../Data.h"
#include "Neuron.h"
#include "Layer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"
#include "InputLayer.h"

template <typename T>
class Network {
private:
    std::vector<Layer> layers;
    double learningRate; // Learning rate
    double testPerformace;

    public:
    Network(std::vector<int> specs, int input_size, int output_size, double learningRate);
    std::vector<double> forward_propagate(const Data<T>& data); //return last layer output
    void back_propagate(const Data<T>& data);
    double activate(const std::vector<double>& neuronWeights, const std::vector<double>& inputsPrevLayer);//dot product
    double transfer(double value); //apply activation function
    double transferDerivative(double value); //calculate derivative of activation function
    void updateWeights(const Data<T>& data);
    int predict(const Data<T>& data);
    void train(int iterrations, const std::vector<Data<T>>& trainingData);
    double test(const std::vector<Data<T>>& testingData);

};


template <typename T>
Network<T>::Network(std::vector<int> specs, int input_size, int numClasses, double learningRate):learningRate(learningRate) {

    for (int i = 0; i < specs.size(); ++i) {
        if(i==0){
            layers.emplace_back(input_size, specs[0]);
        } else {
            layers.emplace_back(specs[i - 1], specs[i]);
        }
    }
    // Output layer
    layers.emplace_back(specs[specs.size() - 1], numClasses);

}

template <typename T>
double Network<T>::activate(const std::vector<double>& neuronWeights, const std::vector<double>& inputsPrevLayer) {
    double sum = neuronWeights.back(); // Start with the bias term
    for (size_t i = 0; i < neuronWeights.size() - 1; ++i) {
        sum += neuronWeights[i] * inputsPrevLayer[i];
    }
    sum += neuronWeights.back(); // Bias term   return sum;
    return sum;
}

template <typename T>
double Network<T>::transfer(double activation){
    return 1.0/(1.0+exp(-activation));
}


template <typename T>
double Network<T>::transferDerivative(double output){
    return output*(1.0-output);
}

template <typename T>
std::vector<double> Network<T>::forward_propagate(const Data<T>& data){
    std::vector<double> inputs = data.get_normalized_feature_vector();//copy of the input vector
    std::vector<double> outputs;
    for(size_t i=0; i<layers.size();i++){
        Layer& layer = layers[i];
       
        for(const Neuron& n:layer.get_neurons()){
            double activation = activate(n.getWeights(), inputs);
            double output = transfer(activation);
            outputs.push_back(output);
        }
        inputs = outputs; // Pass the outputs as the next layer's inputs
        outputs.clear(); // Clear the output vector for the next layer
    }
    return inputs; // Return the outputs of the last layer
}

template <typename T>
void Network<T>::back_propagate(const Data<T>& data) {
    for(size_t i=layers.size()-1; i>0;i--){
        Layer& layer = layers[i];
        std::vector<double> errors;
        std::vector<Neuron>& neurons = layer.get_neurons();
        if(i!=layers.size()-1){
           
            for(int j=0; j< neurons.size();j++){
                double error = 0.0;
                const std::vector<Neuron>& nextLayerNeurons = layers[i+1].get_neurons();
                for(const Neuron& nn :nextLayerNeurons){
                    error += nn.getWeights()[j] * nn.getDelta();
                }
                errors.push_back(error);
            }
        } else {
            // Output layer
            const std::vector<double>& classVector = data.getClassVector();
            for(size_t j=0; j<classVector.size();j++){
              errors.push_back(classVector[j] -  neurons[j].getOutput());
            }
        }

        for(int j=0;j<neurons.size();j++){
            double delta = errors[j] * transferDerivative(neurons[j].getOutput());
            neurons[j].setDelta(delta);
        }
       
    }
}

template <typename T>
void Network<T>::updateWeights(const Data<T>& data){
     std::vector<double> inputs = data.get_normalized_feature_vector();
    for(int i=0;i<layers.size();i++){
        if(i!=0){
            inputs.clear();
            for(const Neuron& n: layers[i-1].get_neurons()){
                inputs.push_back(n.getOutput());
            }
        }

        for(Neuron& n: layers[i].get_neurons()){
            n.updateWeights(inputs, learningRate);
        }
    }
}

template <typename T>
int Network<T>::predict(const Data<T>& data) {
    std::vector<double> outputs = forward_propagate(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

template <typename T>
void Network<T>::train(int iterations, const std::vector<Data<T>>& trainingData) {
    for(int i = 0; i < iterations; ++i) {
        std::cout << "Iteration: " << i + 1 << std::endl;
        double totalError = 0.0;
        for (const Data<T>& data : trainingData) {
            std::vector<double> outputs = forward_propagate(data);
            const std::vector<double>& expectedOutputs = data.getClassVector();
            
            double tempError = 0.0;

            for (size_t j = 0; j < outputs.size(); ++j) {
                double error = expectedOutputs[j] - outputs[j];
                tempError += error * error; // Squared error
            }
            
           totalError += tempError;
            back_propagate(data);
            updateWeights(data);
        }

        std::cout << "Iteration " << i + 1 << ", avg Error: " << totalError / trainingData.size() << std::endl;
    }
}

template <typename T>
double Network<T>::test(const std::vector<Data<T>>& testingData) {
    int correct=0;
    for(const Data<T>& data : testingData) {
        int predicted = predict(data);
        if(predicted == data.get_label()) {
            correct++;
        }
    }
    return static_cast<double>(correct) / testingData.size();
}