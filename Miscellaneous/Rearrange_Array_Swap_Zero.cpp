#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;
class Solution {
    public:
    /* Rearrange an array using swap with 0. */ 

    /* You have two arrays src, tgt, containing two permutations of the numbers 0..n-1. */
    /* You would like to rearrange src so that it equals tgt. The only allowed operations */
    /* is “swap a number with 0”, e.g. {1,0,2,3} -> {1,3,2,0} (“swap 3 with 0”). Write a */
    /* program that prints to stdout the list of required operations. */ 
    static vector<int> rearrangeSwapZero(vector<int> &src, const vector<int> target) {
        //idea, there must be several closed loop in the array, we can first make them into 
        //one big loop, to do it, we use a boolean vector to record where we have found
        long long visited = 0;
        vector<int> res;
        vector<int> map(target.size());
        for (int i = 0; i < target.size(); ++i){
            map[target[i]] = i;
        }
        for (int pos = 0; pos < src.size(); ++pos) {
            if (visited & (1 << pos))   continue;
            visited |= (1 << pos);
            for (int pointer = map[src[pos]]; pointer != pos; pointer = map[src[pointer]]) {
                visited |= (1 << pointer);
            }
            swapZero(src, pos);
            if (pos != 0) res.push_back(pos);
        }
        cout << res.size() << endl;
        //also we need to know that for each value in src, which position it should goes to
        
        while (map[src[0]] != 0) {
            res.push_back(map[src[0]]);
            swapZero(src, map[src[0]]);
        }
        return res;
    }

    //swap zero function
    static void swapZero(vector<int> &nums, int i) {
        int temp = nums[0];
        nums[0] = nums[i];
        nums[i] = temp;
    }



};


int main() {
    vector<int> src({1, 0, 2, 3});
    vector<int> tgt({1, 3, 2, 0});

    vector<int> move = Solution::rearrangeSwapZero(src, tgt);
    cout << "after swap: " << endl;
    for (int num: src) {
        cout << num << " ";
    }
    cout << endl;
    cout << "movement: " << endl;
    for (int m: move) {
        cout << m << " ";
    }
    cout << endl;
}
