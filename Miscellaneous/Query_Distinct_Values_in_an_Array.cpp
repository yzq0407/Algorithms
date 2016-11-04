/* Given an array with n integers, and m queries [low_1, high_1], [low_2, high_2], ..., */ 
/* [low_m, high_m] where low and high are indices in the array, return an array consists */
/* of m integers where each one is the number of distinct values within the corresponding query interval */


//idea:
//1. having a array (last_occur) with the same length, initialize to 0
//2. scan from left to right, if nums[i] is the first occurrence of this value, set the last_occur[i] = 1
//    otherwise, set the last occurrence index to be 0, and last_occur[i] = 1
//4. whenever we see the right end of a query (it will be sorted by high value), sum of number of ones in the last_occur
//   in the same range, and the value should be number of distinct values in that range
//5. to efficiently update and sum the range, using a balanced BST which each node represent one index in the array, and also
//   keeps the sum of all the values in the left tree and itself
//6. when update a idx by value (diff), only need to update log(n) nodes
//7, when sum the range [0, idx], only need to sum up log(n) nodes
//8. total run time O(mlogm + mlogn) (m --> number of queries, n --> number of integers)
#include <vector>
#include <unordered_map>
#include <iostream>

using namespace std;
class Solution {
private:
    typedef pair<int, int> query;

    struct IndexedTreeNode {
        int idx;
        //this field keep the sum of all the values in the left subtree + self
        int sum_left;
        IndexedTreeNode *left;
        IndexedTreeNode *right;
        IndexedTreeNode(int i): idx(i), sum_left(0), left(nullptr), right(nullptr){}

        ~IndexedTreeNode() {
            if (left)   delete left;
            if (right)  delete right;
        }
    };
    
    //build a perfect balanced binary tree with all tree values initialized to 0
    IndexedTreeNode *buildBalancedIndexedTree(int low_index, int high_index) {
        if (low_index > high_index) return nullptr;
        if (low_index == high_index)   return new IndexedTreeNode(low_index);
        //recursively build binary tree
        int mid_idx = low_index + (high_index - low_index) / 2;
        auto root = new IndexedTreeNode(mid_idx);
        root -> left = buildBalancedIndexedTree(low_index, mid_idx - 1);
        root -> right = buildBalancedIndexedTree(mid_idx + 1, high_index);
        return root;
    }

    //update the value of the indexed tree
    void updateIndexedTree(IndexedTreeNode *root, int idx, int diff) {
        if (!root)  return;
        if (root -> idx == idx) {
            root -> sum_left += diff;
            return;
        }
        else if (root -> idx > idx) {
            //idx in the left tree
            root -> sum_left += diff;
            updateIndexedTree(root -> left, idx, diff);
        }
        else {
            //idx in the right tree
            updateIndexedTree(root -> right, idx, diff);
        }
    }

    //get the range sum [0, idx]
    int sumRangeIndexedTree(IndexedTreeNode* root, int idx) {
        if (!root || idx < 0)   return 0;
        if (root -> idx == idx) return root -> sum_left;
        else if (root -> idx < idx) return root -> sum_left + sumRangeIndexedTree(root -> right, idx);
        else return sumRangeIndexedTree(root -> left, idx);
    }

public:
    vector<int> getNumberOfDistinctValues(vector<int>& nums, vector<query> queries) {
        vector<int> res(queries.size());
        //when scan from left to right, store the last occurence index for a given value
        unordered_map<int, int> last_occur;
        //query with index, so that we know where the ans should go
        vector<pair<query, int>> indexedQueries;
        for (int i = 0; i < queries.size(); ++i) {
            indexedQueries.push_back(make_pair(queries[i], i));
        }
        //sort query such that they are ordered by the end value
        sort(indexedQueries.begin(), indexedQueries.end(), [](const pair<query, int>& q1, const pair<query, int>& q2) {
                return q1.first.second < q2.first.second;});
        //build a segment tree for range sum query
        auto root = buildBalancedIndexedTree(0, nums.size() - 1);

        for (int i_nums = 0, i_queries = 0; i_nums < nums.size(); ++i_nums) {
            if (last_occur.find(nums[i_nums]) != last_occur.end()) {
                //this vale has occured before, update the previous location to set it to be 0
                updateIndexedTree(root, last_occur[nums[i_nums]], -1);
            }
            //set the newest location to be 1
            updateIndexedTree(root, i_nums, 1);
            last_occur[nums[i_nums]] = i_nums;
            while (i_queries < indexedQueries.size() && indexedQueries[i_queries].first.second == i_nums) {
                int start = indexedQueries[i_queries].first.first;
                int end = indexedQueries[i_queries].first.second;
                int idx = indexedQueries[i_queries++].second;
                //sum up the range, number of 1s in the range would be number of distinct values
                res[idx] = sumRangeIndexedTree(root, end) - sumRangeIndexedTree(root, start - 1);
            }
        }
        return res;
    }
};


int main() {
    Solution s;
    vector<int> nums = {2, 5, 6, 5, 6, 2, 7, 3, 10, 1, 4, 2};
    vector<pair<int, int>> queries = {{1, 1}, {0, 5}, {2, 4}, {3, 6}, {5, 8}, {1, 8}, {0, 6}}; 
    //result suppose to be {1, 3, 2, 4, 4, 6, 4}
    for (int res: s.getNumberOfDistinctValues(nums, queries)) {
        cout << res << " ";
    }
    cout << endl;
}
