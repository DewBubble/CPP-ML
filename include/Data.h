#pragma once
#include <vector>
#include <cstdint>
#include <memory>


template <typename T>
class Data{
    std::vector<T> feature_vector;
    std::vector<double> normalized_feature_vector;
    std::vector<double> normalized_feature_vector_mean;
    std::vector<double> class_vector;
    uint8_t label;
    int enum_label;
public:
    Data(const std::vector<T>& v);
    Data(std::vector<T>&& v);

    void set_label(uint8_t);
    void set_enum_label(int);


    uint8_t get_label() const;
    int get_enum_label() const;
    const std::vector<T>& get_feature_vector()const;
    const std::vector<double>& get_normalized_feature_vector() const ;
    void normalize_feature_vector(std::vector<double>& mean, std::vector<double>& stddev);
    void setClassVector(int classCount);
    const std::vector<double>& getClassVector() const;


};

template <typename T>
Data<T>::Data(std::vector<T>&& v): feature_vector(std::move(v)) {};

template <typename T>
Data<T>::Data(const std::vector<T>& v): feature_vector(std::move(v)) {};

template <typename T>
void Data<T>::set_label(uint8_t value) {
    label = value;
};

template <typename T>
void Data<T>::set_enum_label(int value) {
    enum_label = value;
};


template <typename T>
uint8_t Data<T>::get_label () const{
    return label;
};


template <typename T>
int Data<T>::get_enum_label() const{
    return enum_label;
};

template <typename T>
const std::vector<T>& Data<T>::get_feature_vector() const{
    return feature_vector;
};

template <typename T>
void Data<T>::normalize_feature_vector(std::vector<double>& mean, std::vector<double>& stddev) {
    for (size_t i = 0; i < feature_vector.size(); ++i) {
        double normalized_value = (static_cast<double>(feature_vector[i]) - mean[i]) / stddev[i];
        normalized_feature_vector.push_back(normalized_value);
        normalized_feature_vector_mean.push_back(mean[i]);
    }
};

template <typename T>
const std::vector<double>& Data<T>::get_normalized_feature_vector() const {
    return normalized_feature_vector;
}

template <typename T>
void Data<T>::setClassVector(int classCount) {
    for(int i=0;i<classCount; ++i) {
        if(label == i) {
            class_vector.push_back(1);
        } else {
            class_vector.push_back(0);
        }
    }
}

template <typename T>
const std::vector<double>& Data<T>::getClassVector() const{
    return class_vector;
}
