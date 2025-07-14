#pragma once
#include <random>
#include <functional>
static auto createRandomNumberGenerator(double min, double max) {
    std::random_device rd;
    std::mt19937 engine{rd()}; // random number generator
    std::uniform_real_distribution<double> distribution{min, max}; // uniform distribution for random number generation

    return [distribution, engine]() mutable {
        return distribution(engine);
    }; // generate a random number using the uniform distribution
}

static  auto gen0To1 = createRandomNumberGenerator(0.0, 1.0);