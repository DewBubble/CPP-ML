#pragma once
#include <vector>
#include <map>
#include "Data.h"
#include <random>

template <typename T>
struct Cluster{
    std::vector<double> centroid;
    std::vector<size_t> pointIndexes; // indices of points in the training data
    std::map<int, int> class_count; // class counts for the points in this cluster
    int most_frequent_class = -1; // most frequent class in the cluster, initialized to -1
   
    Cluster(const Data<T>& dataPoint, size_t index) {
        for(T value : dataPoint.get_feature_vector()) {
            centroid.push_back(static_cast<double>(value)); // initialize centroid with feature vector values
        }
       
        pointIndexes.push_back(size_t);
        class_count[dataPoint.get_label()]=1; // initialize class count with the label of the first point);
        most_frequent_class = dataPoint.get_label();
    }

    // Add a point to the cluster
    void add_point(const Data<T>& dataPoint, size_t index) {
        pointIndexes.push_back(index);
        for(size_t i = 0; i < centroid.size(); ++i) {
            int oldsize = centroid.size();
            double v = centroid[i]*oldsize; // scale centroid value
            centroid[i] = (v + static_cast<double>(dataPoint.get_feature_vector()[i])) / (oldsize + 1); // update centroid}
        }
        if(class_count.find(dataPoint.get_label()) != class_count.end()) {
            class_count[dataPoint.get_label()]++;
            if(class_count[dataPoint.get_label()] > class_count[most_frequent_class]) {
                most_frequent_class = dataPoint.get_label();
            }
        } else {
            class_count[dataPoint.get_label()] = 1;
        }
        
    }

    int get_most_frequent_class() const {
        return most_frequent_class; // return the most frequent class in the cluster
    }

    void reset() {
        pointIndexes.clear();
        class_count.clear();
        most_frequent_class = -1;
    }
   

};

template <typename T>
class kmeans{
    int k; // number of clusters
    std::vector<Cluster<T>> clusters; // vector of clusters
    std::mt19937 engine(std::random_device{}()); // random number generator
    std::uniform_real_distribution<double> distribution(0.0,1.0); // uniform distribution for random number generation

    public:
    kmeans(int k_value) : k(k_value) {};
    void initialize_clusters(const std::vector<Data<T>>& trainingData);
    void train(const std::vector<Data<T>>& trainingData, const std::vector<Data<T>>& validationData,int maxIterations = 10);
    double validate(const std::vector<Data<T>>& validationData);
    void test(const std::vector<Data<T>>& testData)
    double generateRandomNumber() {
        return distribution(engine); // generate a random number using the uniform distribution
    }

    double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for(size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum); // return the Euclidean distance
    }
};

template <typename T>
void kmeans<T>::initialize_clusters(const std::vector<Data<T>>& trainingData) {
    std::unordered_set<size_t> usedIndices; // to keep track of already selected indices
    for(size_t i = 0; i < k; ++i) {
        size_t randomIndex = -1;
        while(usedIndices.find(randomIndex) != usedIndices.end()) {
            randomIndex = static_cast<size_t>(generateRandomNumber() * trainingData.size());
        }
        clusters.emplace_back(trainingData[randomIndex], randomIndex); // create a new cluster with a random data point
    }
}

template <typename T>
void kmeans<T>::train(const std::vector<Data<T>>& trainingData, const std::vector<Data<T>>& validationData, int maxIterations) {
    std::unordered_set<size_t> usedIndices; // to keep track of already selected indices
    double lastAccuracy = -1; // variable to store accuracy
    for(int iter = 0;iter<maxIterations; ++iter){
        for(size_t i = 0; i < trainingData.size(); ++i) {
            size_t randomIndex = -1;
            while(usedIndices.find(randomIndex) != usedIndices.end()) {
                randomIndex = static_cast<size_t>(generateRandomNumber() * trainingData.size());
            }
            usedIndices.insert(randomIndex);
            double minDistance = std::numeric_limits<double>::max();
            size_t closestClusterIndex = -1;
            for(size_t j = 0; j < k; ++j) {
                double distance = calculateDistance(trainingData[randomIndex].get_feature_vector(), clusters[j].centroid);
                if(distance < minDistance) {
                    minDistance = distance;
                    closestClusterIndex = j;
                }
            }
            if(closestClusterIndex != -1) {
                clusters[closestClusterIndex].add_point(trainingData[randomIndex], randomIndex);
            }
        }
        double accuracy = validate(validationData); // validate the clusters after each iteration
        if(lastAccuracy > 0){
            if(std::abs(lastAccuracy- accuracy) < 0.01) {
                std::cout << "Stopping early at iteration " << iter << " due to minimal change in accuracy.\n";
                break; // stop training if accuracy does not change significantly
            }
        }
        lastAccuracy = accuracy; // update lastAccuracy for the next iteration
        usedIndices.clear(); // clear used indices for the next iteration
        for(size_t i = 0; i < k; ++i) {
            clusters[i].reset(); // reset clusters for the next iteration
        }
        
    }
}

template <typename T>
double kmeans<T>::validate(const std::vector<Data<T>>& validationData) {
    int correctClassifications = 0;
    for(size_t i = 0; i < validationData.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        size_t closestClusterIndex = -1;
        for(size_t j = 0; j < clusters.size(); ++j) {
            double distance = calculateDistance(validationData[i].get_feature_vector(), clusters[j].centroid);
            if(distance < minDistance) {
                minDistance = distance;
                closestClusterIndex = j;
            }
        }
        if(clusters[closestClusterIndex].get_most_frequent_class() == validationData[i].get_label()){
            // Increment the correct classification count
            ++correctClassifications;
        }
    }
    double accuracy = static_cast<double>(correctClassifications) / validationData.size();
    std::cout<<"Validation accuracy: " << std::to_string(accuracy * 100) + "%\n");
    return accuracy;

}

template <typename T>
void kmeans<T>::test(const std::vector<Data<T>>& testData) {
    int correctClassifications = 0;
    for(size_t i = 0; i < testData.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        size_t closestClusterIndex = -1;
        for(size_t j = 0; j < clusters.size(); ++j) {
            double distance = calculateDistance(testData[i].get_feature_vector(), clusters[j].centroid);
            if(distance < minDistance) {
                minDistance = distance;
                closestClusterIndex = j;
            }
        }
        if(clusters[closestClusterIndex].get_most_frequent_class() == testData[i].get_label()){
            // Increment the correct classification count
            ++correctClassifications;
        }
    }
    double accuracy = static_cast<double>(correctClassifications) / testData.size();
    std::cout<<"Test accuracy: " << std::to_string(accuracy * 100) + "%\n";
    return accuracy;
}


