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



int main() {
    std::vector<std::vector<int>> test ({{1, 2, 3, 4, 5, 6}, {2, 4, 5, 6, 7, 8}});
    std::cout << test;
}
