#pragma once
#include <fstream>
#include <stdint.h>
#include <vector>
#include <cstdint>
#include "data.h"
#include <vector>
#include <string>
#include <unordered_map>

#include "../include/Data.h"
#include "Neighbor.h"

#include <algorithm>
#include <random>
#include <iostream>
#include "../include/Neighbor.h"

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

    
    const std::vector<Data<T>>& get_training_data() const;
    const std::vector<Data<T>>& get_test_data() const;
    const std::vector<Data<T>>& get_validation_data() const;
    void normalize_feature_vector();
    int get_num_classes() const {
        return num_classes;
    }

};

template <typename T>
void DataHandler<T>::normalize_feature_vector() {

    std::vector<double> mean;
    std::vector<double> stddev;
    int dimensions = data_array[0].get_feature_vector().size();
    mean.resize(dimensions, 0.0);
    stddev.resize(dimensions, 0.0);
    for (Data<T>& data: data_array) {
        for(int i=0;i<dimensions; ++i) {
            mean[i] +=data.get_feature_vector()[i];
            stddev[i] +=data.get_feature_vector()[i] * data.get_feature_vector()[i];
        }
    }
    for(int i=0; i < mean.size(); ++i) {
        mean[i] /= data_array.size();
        stddev[i] = sqrt(stddev[i] / data_array.size() - mean[i] * mean[i]);
    }



    for(auto& data : data_array) {
        data.normalize_feature_vector(mean, stddev);
        data.setClassVector(num_classes);
    }
};
template <typename T>
void DataHandler<T>::read_feature_vector(std::string filename){
    uint32_t header[4]; // 0: magic number, 1: number of images, 2: number of rows, 3: number of columns
    T bytes[4];

    std::ifstream  fs(filename, std::ios::binary);
    if(fs.is_open()){
        for(int i=0;i<4;i++){
            if(fs.read(reinterpret_cast<char*>(bytes), 4)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout<<"done reading feature header\n";
        
        
        feature_vector_size = header[2]*header[3];
        for(int i=0;i<header[1];i++){
            std::vector<T> imageBytes (feature_vector_size);
            if(fs.read(reinterpret_cast<char*>(imageBytes.data()), feature_vector_size)){
                data_array.push_back(Data<uint8_t>(imageBytes));
            }
            else{
                 std::cout<<"error reading feature vector\n";
                break;
            };    
        }  
        std::cout<<"done reading image\n";


    }else{
        std::cout<<"error reading image file "<< filename<<std::endl;
        return;
    }  

};
template <typename T>
void DataHandler<T>::read_feature_label(std::string filename){
    uint32_t header[2]; // 0: magic number, 1: number of images
    uint8_t bytes[4];

    std::ifstream fs(filename, std::ios::binary);
    if(fs.is_open()){
        for(int i=0;i<2;i++){
            if(fs.read(reinterpret_cast<char*>(bytes), 4)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
       std::cout<<"done reading label header\n";
                
        for(int i=0;i<header[1];i++){
            uint8_t label[1];
            if(fs.read(reinterpret_cast<char*>(label), 1)){
                data_array[i].set_label(label[0]);
//                data_array[i]->set_enum_label(class_map->at(label[0]));
            }
            else{
                std::cout<<"error reading label\n";
            }
        }  
        std::cout<<"done reading label\n";


    }else{
       std::cout<<"error reading label file "<< filename<<std::endl;
        return;
    }  
};
template <typename T>
 void DataHandler<T>::split_data(){
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data_array.begin(),data_array.end(), g);
    int training_size = data_array.size() * TRAINING_DATA_PERCENTAGE;
    int validation_size = data_array.size() * VALIDATION_DATA_PERCENTAGE;
    int test_size = data_array.size() -training_size - validation_size;

    training_data.assign(data_array.begin(), data_array.begin() + training_size);
    validation_data.assign(data_array.begin() + training_size, data_array.begin() + training_size + validation_size);
    test_data.assign(data_array.begin() + training_size + validation_size, data_array.end());
 };

template <typename T>
 void DataHandler<T>::count_classes(){
    num_classes = 0;
    for(size_t i=0;i<data_array.size();i++){
        auto iter = class_map.find(data_array[i].get_label());
        if(iter == class_map.end()){
            data_array[i].set_enum_label(num_classes);
            class_map[ data_array[i].get_label()] = num_classes;
            num_classes++;
        }else{
            data_array[i].set_enum_label(iter->second);
        }
    }
    std::cout<<"num_classes: "<< num_classes << std::endl;
 }

 template <typename T>
 uint32_t DataHandler<T>::convert_to_little_endian(const unsigned char* bytes){
    return (static_cast<uint32_t>(bytes[0])<<24) |
           (static_cast<uint32_t>(bytes[1])<<16) |
           (static_cast<uint32_t>(bytes[2])<<8)  |
           (static_cast<uint32_t>(bytes[3]));
 }

template <typename T>
const std::vector<Data<T>>& DataHandler<T>::get_training_data() const{
    return training_data;
};

template <typename T>
const std::vector<Data<T>>& DataHandler<T>::get_test_data() const{
    return test_data;
};

template <typename T>
const std::vector<Data<T>>& DataHandler<T>::get_validation_data() const{
    return validation_data;
};

