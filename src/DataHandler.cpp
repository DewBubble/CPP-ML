#include "../include/DataHandler.h"

#include <print>
#include <algorithm>
#include <random>
#include <iostream>

template <typename T>
void DataHandler<T>::read_feature_vector(std::string filename){
    uint32_t header[4]; // 0: magic number, 1: number of images, 2: number of rows, 3: number of columns
    unsigned char bytes[4];

    FILE *f = fopen(filename.c_str(), "rb");
    if(f){
        for(int i=0;i<4;i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout<<"done reading feature header\n";
        
        
        feature_vector_size = header[2]*header[3];
        for(int i=0;i<header[1];i++){
            std::vector<T> imageBytes (feature_vector_size);
            if(fread(imageBytes, sizeof(uint8_t), feature_vector_size, f)){
                data_array.push_back(std::make_shared<Data>(imageBytes, feature_vector_size));
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
    unsigned char bytes[4];

    FILE *f = fopen(filename.c_str(), "r");
    if(f){
        for(int i=0;i<2;i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
       std::cout<<"done reading label header";
                
        for(int i=0;i<header[1];i++){
            uint8_t label[1];
            if(fread(label, sizeof(label), 1, f)){
                data_array[i]->set_label(label[0]);
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
        auto iter = class_map.find(data_array[i]->get_label());
        if(iter == class_map.end()){
            data_array[i]->set_enum_label(num_classes);
            class_map[ data_array[i]->get_label()] = num_classes;
            num_classes++;
        }else{
            data_array[i]->set_enum_label(iter->second);
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
std::vector<Data<T>>& DataHandler<T>::get_training_data() const{
    return training_data;
};

template <typename T>
std::vector<Data<T>>& DataHandler<T>::get_test_data() const{
    return test_data;
};

template <typename T>
std::vector<Data<T>>& DataHandler<T>::get_validation_data() const{
    return validation_data;
};

int main() {
    DataHandler<uint8_t> handler;
    handler.read_feature_vector("../train-images.idx3-ubyte");
    handler.read_feature_label("../train-labels.idx1-ubyte");
    handler.split_data();
    handler.count_classes();
}