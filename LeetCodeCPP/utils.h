#ifndef ALGO_UTILS
#define ALGO_UTILS
#include <iostream>
#include <vector>

//allow us to print vector of printables
//
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> vals) {
    for (const T& val: vals) {
        os << val << " ";
    }
    return os;
}


#endif

