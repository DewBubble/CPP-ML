#include "../include/Data.h"

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
std::vector<T>& Data<T>::get_feature_vector() const{
    return feature_vector;
};

