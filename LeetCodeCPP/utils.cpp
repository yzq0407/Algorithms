#include <iostream>
#include <vector>
#include "utils.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> vals) {
    for (auto val: vals) {
        os << val << " ";
    }
    std::cout <<std::endl;
    return os;
}

