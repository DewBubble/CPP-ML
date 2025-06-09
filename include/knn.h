#include<vector>
#include "DataHandler.h"#
#include <cmath>
#include <set>


bool NeighborCompare(const Neighbor& a, const Neighbor& b) {
    return a.distance < b.distance; //accending order
};

template <typename T>
class knn{
    int k
    public:
    knn(int k_value) : k(k_value) {};


    std::set<Neighbor,  decltype(NeighborCompare)> find_nearest_neighbors(const Data<T>& queryPoint, const std::vector<Data<T>>& traingingData);
    int predict(const Data<T>& queryPoint, const std::vector<Data<T>>& traingingData);
    double calculate_distance(const Data<T>& point1, const Data<T>& point2);
    double validata();
    double test();

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
std::set<Neighbor,  decltype(NeighborCompare)> knn<T>::find_nearest_neighbors(const Data<T>& queryPoint, const std::vector<Data<T>>& traingingData) {
    std::set<Neighbor, delcltype(NeighborCompare)>& neighbors;
    // Assuming 'data' is a member variable containing the training dataset
    for (const auto& point : traingingData) {
        double distance = calculate_distance(point, queryPoint);
        Neighbor neighbor{distance, point.get_enum_label()};
        neighbors.insert(neighbor);
        if(neighbors.size(>K)
        {
            neighbors.erase(std::prev(neighbors.end())); // remove the farthest neighbor
        } // neighbors.size(>K)
        )
    }
    return neighbors;
}

template <typename T>
int knn<T>::predict(const Data<T>& queryPoint, const std::vector<Data<T>>& traingingData){
    auto neighbors = find_nearest_neighbors(queryPoint, training_data);
    std::unordered_map<int, int> class_count;

    int max_count = 0;
    int index =0;
    for(const auto& neighbors :neighbors){
        class_count[neighbors.index]++;
        if(max_count < class_count[neighbors.index]){
            max_count = class_count[neighbors.index];
            index = neighbors.index;
        }
    }
    return traingingData[index].get_label();
   
}