#include "../include/knn.h"
#include<iostream>
#include "../include/kmeans.h"

int main() {
    DataHandler<uint8_t> handler;
    handler.read_feature_vector("../train-images.idx3-ubyte");
    handler.read_feature_label("../train-labels.idx1-ubyte");
    handler.split_data();
    handler.count_classes();

    // knn<uint8_t> classifier(5); // Example with k=5
    // double rate = classifier.test(handler.get_test_data(), handler.get_training_data());

    
   
    kmeans<uint8_t> classifier(100); 
    std::cout << "init kmean classifiler "<< std::endl;
    classifier.initialize_clusters(handler.get_training_data());
    std::cout << "start training kmean classifier "<< std::endl;
    classifier.train(handler.get_training_data(), handler.get_validation_data(), 50);
    std::cout << "start testing kmean classifier "<< std::endl;
    classifier.test(handler.get_test_data());
    // std::cout << "Test accuracy: " << rate * 100 << "%" << std::endl;
    return 0;
}