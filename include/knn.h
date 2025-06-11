#include<vector>
#include "DataHandler.h"
#include <cmath>
#include <set>
#include <iostream>
#define EUCLID

struct NeighborCompare {
    bool operator()(const Neighbor& a, const Neighbor& b) const{
        return a.distance < b.distance; //accending order
    }
};

template <typename T>
class knn{
    int k;

    public:
    knn(int k_value) : k(k_value) {};

    std::set<Neighbor,  NeighborCompare> find_nearest_neighbors(const Data<T>& queryPoint, const std::vector<Data<T>>& traingingData);
    int predict(const Data<T>& queryPoint, const std::vector<Data<T>>& traingingData);
    double calculate_distance(const Data<T>& point1, const Data<T>& point2);
    double test(const std::vector<Data<T>>& testData, const std::vector<Data<T>>& trainingData);

};

template <typename T>
double knn<T>::calculate_distance(const Data<T>& point1, const Data<T>& point2) {
    double distance = 0.0;
    const auto& vec1 = point1.get_feature_vector();
    const auto& vec2 = point2.get_feature_vector();
    #ifdef EUCLID
        for (size_t i = 0; i < vec1.size(); ++i) {
            distance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
        }
    #elif defined MANHATTAN
        for(size_t i=0;i<vec1.size();i++){
            distance += std::abs(vec1[i] - vec2[i]);
        }
    #endif
    return std::sqrt(distance);
};



template <typename T>
std::set<Neighbor,  NeighborCompare> knn<T>::find_nearest_neighbors(const Data<T>& queryPoint, const std::vector<Data<T>>& traingingData) {
    std::set<Neighbor, NeighborCompare> neighbors;
    // Assuming 'data' is a member variable containing the training dataset
    for (size_t i = 0; i < traingingData.size(); ++i) {

        double distance = calculate_distance(traingingData[i], queryPoint);
        Neighbor neighbor{distance, i};
        neighbors.insert(neighbor);
        if(neighbors.size()>k)
        {
            neighbors.erase(std::prev(neighbors.end())); // remove the farthest neighbor
        } // neighbors.size(>K)

    }
    return neighbors;
}

template <typename T>
int knn<T>::predict(const Data<T>& queryPoint, const std::vector<Data<T>>& trainingData){
    auto neighbors = find_nearest_neighbors(queryPoint, trainingData);
    std::unordered_map<int, int> class_count;

    int max_count = 0;
    int label =0;
    for(const auto& neighbors :neighbors){
        int neighborLabel = trainingData[neighbors.index].get_label();
        class_count[neighborLabel]++;
        if(max_count < class_count[neighborLabel]){
            max_count = class_count[neighborLabel];
            label = neighborLabel;
        }
    }
    return label;
   
}

template <typename T>
double knn<T>::test(const std::vector<Data<T>>& testData, const std::vector<Data<T>>& trainingData) {
    int correct_predictions = 0;
    for(const auto& point : testData){
        int predicted_label = predict(point, trainingData);
        std::cout<< "Predicted label: " << predicted_label << ", Actual label: " << static_cast<int>(point.get_label()) << std::endl;
        if(predicted_label == point.get_label()){
            correct_predictions++;
        }
    }
    return static_cast<double>(correct_predictions) / testData.size();
    
}

