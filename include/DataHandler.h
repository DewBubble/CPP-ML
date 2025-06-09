#pragma once
#include <fstream>
#include <stdint.h>
#include <vector>
#include <cstdint>
#include "data.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "../include/Data.h"
#include "Neighbor.h"



template <typename T>
class DataHandler{
    std::vector<Data<T>> data_array;
    std::vector<Data<T>> training_data;
    std::vector<Data<T>> test_data;
    std::vector<Data<T>> validation_data;

    int num_classes;
    int feature_vector_size;
    std::unordered_map<uint8_t, int> class_map;

    const double TRAINING_DATA_PERCENTAGE = 0.75;
    const double VALIDATION_DATA_PERCENTAGE = 0.20;
    const double TEST_DATA_PERCENTAGE = 0.05;

    public:
 

    void read_feature_vector(std::string filename);
    void read_feature_label(std::string filename);

    void split_data();
    void count_classes();
    uint32_t convert_to_little_endian(const unsigned char* bytes);

    
    std::vector<Data<T>>& get_training_data() const;
    std::vector<Data<T>>& get_test_data() const;
    std::vector<Data<T>>& get_validation_data() const;
};