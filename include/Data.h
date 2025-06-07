#pragma once
#include <vector>
#include <cstdint>
#include <memory>


template <typename T>
class Data{
    std::vector<T> feature_vector;
    uint8_t label;
    int enum_label;
public:
    Data(const std::vector<T>& v);
    Data(std::vector<T>&& v);

    void set_label(uint8_t);
    void set_enum_label(int);


    uint8_t get_label() const;
    int get_enum_label() const;
    std::vector<T>& get_feature_vector()const;
};