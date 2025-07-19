#include "../include/knn.h"
#include<iostream>
#include "../include/kmeans.h"
#include "../include/nn/Network.h"

int main() {
    DataHandler<uint8_t> handler;
    handler.read_feature_vector("C:/CppProjects/ETL/train-images.idx3-ubyte");
    handler.read_feature_label("C:/CppProjects/ETL/train-labels.idx1-ubyte");
    handler.count_classes();
    handler.normalize_feature_vector();
    handler.split_data();
    
    



    // knn<uint8_t> classifier(5); // Example with k=5
    // double rate = classifier.test(handler.get_test_data(), handler.get_training_data());
   
    //kmeans<uint8_t> classifier(100); 
    std::vector<int> specs = {50}; // Example layer sizes
    Network<uint8_t> nn(specs, handler.get_training_data()[0].get_feature_vector().size(), handler.get_num_classes(), 0.001);

    std::cout << "start training nn classifier "<< std::endl;
    nn.train(50, handler.get_training_data());
    std::cout << "start testing nn classifier "<< std::endl;
    double rate = nn.test(handler.get_test_data());
    std::cout << "Test accuracy: " << rate * 100 << "%" << std::endl;
    return 0;
}