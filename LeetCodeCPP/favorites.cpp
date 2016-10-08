#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include "accessories.h"


using namespace std;
//contains all the solution functions for the favoriates questions
class Solution {
    /* You have a number of envelopes with widths and heights given as a pair of integers (w, h). */
    /* One envelope can fit into another if and only if both the width and height of one envelope */ 
    /* is greater than the width and height of the other envelope. */
    /* What is the maximum number of envelopes can you Russian doll? (put one inside other) */
    /* https://leetcode.com/problems/russian-doll-envelopes/ */
    int maxEnvelopes(vector<pair<int, int>>& envelopes) { 
        //basic approach is to sort the envelop by their width
        //if their width is equal, put the one with larger height in the front
        //in this way we can use the same algorithm of longest increasing subsequence 
        sort(envelopes.begin(), envelopes.end(), [](const pair<int, int> &p1, const pair<int, int> &p2) {
                return p1.first != p2.first? p1.first < p2.first: p1.second > p2.second;});
        //this vector dynamically keep the tail with different length
        vector<int> tails;
        for (auto p: envelopes) {
            if (tails.size() == 0 || tails[tails.size() - 1] < p.second)   tails.push_back(p.second);
            else {
                //trigger binary search to find the first element that is LARGER than or equal to p.second
                int low = 0;
                int high = tails.size() - 1;
                //keep invariant that tails[high] >= p.second
                while (low <= high) {
                    int mid = (low + high) >> 1;
                    if (tails[mid] >= p.second) high = mid - 1;
                    else    low = mid + 1;
                }
                tails[high + 1] = p.second;
            }
        }
        //the tails size must be the length of increasing subsequence
        return tails.size();
    }

/*     You are given an integer array nums and you have to return a new counts array. */ 
/*     The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i]. */
/*     https://leetcode.com/problems/count-of-smaller-numbers-after-self/ */
    vector<int> countSmaller(vector<int> &nums) {
        //this problem can be approached by two different algorithms
        //one is using BIT (binary indexed tree), which guarantees log(n) time
        //basic idea is to dicretize the whole array and map them into (0 -- n - 1) based on their position in the sorted list
        //then we start from the right most number in the array, count the sum in BIT, then update the BIT
        //-----------------------------------------------------------------------------------------------
        //implementation
        //basically we want to get a argsort of the number, ex [3, 1, 2] --->   [2, 0, 1]
        vector<int> argsort(nums.size(), 0);
        for (int i = 0; i < nums.size(); ++i)   argsort[i] = i;
        //use lambda to sort the args, thanks c++11!! (remember the tie situation, where nums[i] == nums[j], we want the stable sorting
        //or we can use std::stable_sort)
        stable_sort(argsort.begin(), argsort.end(), [&nums](int i, int j) {return nums[i] < nums[j];});
        //then based on the arg sort we can get the discretization, discret[i] = j means nums[j] ranks i th in the whole array
        vector<int> discret(nums.size(), 0);
        for (int i = 0; i < nums.size(); ++i)   discret[argsort[i]] = i;
        //start from the rightmost most number
        vector<int> b_indexed_tree(nums.size() + 1, 0);
        for (int i = nums.size() - 1; i >= 0; --i) {
            //its index in bit is nums.size() - i
            int idx = discret[i];
            argsort[i] = sum_bit(b_indexed_tree, idx);
            update_bit(b_indexed_tree, idx + 1, 1);
        }
        //we should name the return value to be argsort, though, it is a save of space
        return argsort;
    }

    //helper functions to update and sum binary indexed tree
    int sum_bit(vector<int> &bit, int idx) {
        int res = 0;
        while (idx != 0) {
            res += bit[idx];
            idx -= (idx & -idx);
        }
        return res;
    }

    void update_bit(vector<int> &bit, int idx, int diff) {
        while (idx < bit.size()) {
            bit[idx] += diff;
            idx += (idx & -idx);
        }
    }

    //------run time 39ms, beats 72.80% -----------------

/*     this is the second implementation of countSmaller, now we have created another datastructure */ 
/*     called IndexedTreeNode (see in the header file). By using this, we can have a O(nlogn) amortized running time */
/*     but in the worst case, it gives O(n^2) running time */
    vector<int> countSmaller2(vector<int> &nums) {
        vector<int> res(nums.size(), 0);
        IndexedTreeNode *root = nullptr;
        for (int i = nums.size() - 1; i >= 0; --i) {
            int smaller = 0;
            //recursively put the node into the tree, at the same time, count the sum of smaller
            root = put(root, nums[i], smaller);
            res[i] = smaller;
        }
        //no memory leak!
        cleanup(root);
        return res;
    }

    IndexedTreeNode* put(IndexedTreeNode *root, int val, int &smaller) {
        if (!root)  return new IndexedTreeNode(val);
        //three cases
        if (root -> val == val) {
            ++root -> duplicates;
            smaller += root -> smaller;
        }
        else if (root -> val < val) {
            smaller += (root -> duplicates + root -> smaller);
            root -> right = put(root -> right, val, smaller);
        }
        else {
            ++root -> smaller;
            root -> left = put(root -> left, val, smaller);
        }
        return root;
    }

    void cleanup (IndexedTreeNode *root) {
        if (!root)  return;
        cleanup(root -> left);
        cleanup(root -> right);
        delete(root);
    }

    //----running time 33ms, beats 89.18%------//






};

/*     308 Range Sum Query 3D - Mutable */
/*     Given a 2D matrix matrix, find the sum of the elements inside the rectangle defined by its upper left corner (row1, col1) and lower right */ 
class NumMatrix {
    // this is a good example of using BIT to efficiently update and query the range sum, the basic formula to query is 
    // sum(xmin, ymin, xmax, ymax) = rangeSum(0, 0, xmax, ymax) + rangeSum(0, 0, xmin - 1, ymin - 1) - rangeSum(0, 0, xmax, ymin) - rangeSum(0, 0, xmin, ymax)
    // also we need to keep the numbers of original matrix for update

    public:
        NumMatrix(vector<vector<int>> &matrix) {
            if (matrix.size() == 0) return;
            n = matrix.size();
            m = matrix[0].size();
            this->matrix = vector<vector<int>>(n, vector<int>(m, 0));
            two_d_bit = vector<vector<int>>(n + 1, vector<int>(m + 1, 0));
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    update(i, j, matrix[i][j]);
                }
            }
        }

        void update(int row, int col, int val) {
            int diff = val - matrix[row][col];
            matrix[row][col] += diff;
            for (int i = row + 1; i < two_d_bit.size(); i += (i & -i)) {
                for (int j = col + 1; j < two_d_bit[0].size(); j += (j & -j)) {
                    two_d_bit[i][j] += diff;
                }
            }
        }

        int sumRegion(int row1, int col1, int row2, int col2) {
            //using the formula shown above
            return sumRegion(row2, col2) + sumRegion(row1 - 1, col1 - 1) - sumRegion(row1 - 1, col2) - sumRegion(row2, col1 - 1);
        }
        
        //this function overload the previous one by taking row and col function to calculate
        //rangeSum(0, 0, row, col)
        int sumRegion(int row, int col) {
            int sum = 0;
            for (int i = row + 1; i != 0; i -= (i & -i)) {
                for (int j = col + 1; j != 0; j -= (j & -j)) {
                    sum += two_d_bit[i][j];
                }
            }
            return sum;
        }

    private:
        int n;
        int m;
        vector<vector<int>> matrix;
        vector<vector<int>> two_d_bit;

    //--------run time 62ms -- beats 39.42% --------//
};

//Note that this question can be solved using quadtree approach as well (though much slower)
//Quadtree is another interesting topic, which I have several interesting examples in miscellaneous folder
//Here is the implementation

class NumMatrix_Quad {
    //this is the basic QtreeNode
    struct QTreeNode {
        static constexpr int branches = 4;
        int sum;
        vector<QTreeNode*> children;
        const pair<int, int> leftTop;
        const pair<int, int> rightBot;

        //constructor
        QTreeNode(const pair<int, int>& lt, const pair<int, int>& rb, int v):
            sum(v), leftTop(lt), rightBot(rb), children(branches, nullptr) {
            }
    };


    public:
        NumMatrix_Quad(vector<vector<int>> &matrix): root(nullptr) {
            if (matrix.size() == 0) return;
            n = matrix.size();
            m = matrix[0].size();
            root = constructQTree(matrix, make_pair(0, 0), make_pair(n - 1, m - 1));
        }

        void update(int row, int col, int val) {
            updateQTree(root, row, col, val);
        }

        int sumRegion(int row1, int col1, int row2, int col2) {
            return sumQTree(root, make_pair(row1, col1), make_pair(row2, col2));
        }

        ~NumMatrix_Quad() {
            cleanup(root);
        }

    private:
        QTreeNode* root;
        int n;
        int m;

        QTreeNode* constructQTree(const vector<vector<int>> &matrix, const pair<int, int> &lt, const pair<int, int> &rb) {
            if (lt.first > rb.first || lt.second > rb.second)   return nullptr;
            if (lt.first == rb.first && lt.second == rb.second) return new QTreeNode(lt, rb, matrix[lt.first][lt.second]);
            QTreeNode* node = new QTreeNode(lt, rb, 0);
            //otherwise setup its four children
            int xmid = (lt.first + rb.first) >> 1;
            int ymid = (lt.second + rb.second) >> 1;
            //left top child
            node->children[0] = constructQTree(matrix, make_pair(lt.first, lt.second), make_pair(xmid, ymid));
            
            //left bot child
            node->children[1] = constructQTree(matrix, make_pair(xmid + 1, lt.second), make_pair(rb.first, ymid));
            //right top child
            node->children[2] = constructQTree(matrix, make_pair(lt.first, ymid + 1), make_pair(xmid, rb.second));
            //right bot child
            node->children[3] = constructQTree(matrix, make_pair(xmid + 1, ymid + 1), make_pair(rb.first, rb.second));
            for (int i = 0; i < QTreeNode::branches; ++i) {
                if (node->children[i]) {
                    node->sum += node->children[i]->sum;
                }
            }
            return node;
        }

        //this function efficiently search for the right node to update the value of the element
        //it returns the value diff so that all the parents must change the value as well
        int updateQTree(QTreeNode* node, int i, int j, int val) {

            if (!node)  return 0;
            auto lt = node -> leftTop;
            auto rb = node -> rightBot;
            //a single node, if our calculation is correct, it must be the right node
            if (lt.first == rb.first && lt.second == rb.second) {
                int diff = val - node -> sum;
                node -> sum = val;
                return diff;
            }
            //otherwise just search for the right child to be updated
            int diff = 0;
            bool left = j <= ((lt.second + rb.second) >> 1)? true: false;
            bool up = i <= ((lt.first + rb.first) >> 1)? true: false;
            if (left) 
                diff = up? updateQTree(node->children[0], i, j, val): updateQTree(node->children[1], i, j, val);
            else
                diff = up? updateQTree(node->children[2], i, j, val): updateQTree(node->children[3], i, j, val);
            node -> sum += diff;
            return diff;

        }

        //this function efficiently calculate the sum of the region based on the QTreeNode
        //we return value if and only if 1. node is null 2. node boundaries is strictly smaller or equal to the boundaries 
        //specified by the pair
        int sumQTree(const QTreeNode* node, const pair<int, int>& lt, const pair<int, int>& rb) {
            if (!node)  return 0;
            const auto lt_node = node -> leftTop;
            const auto rb_node = node -> rightBot;
            if (rb.first < lt.first || rb.second < lt.second)   return 0;
            if (lt_node.first >= lt.first && lt_node.second >= lt.second && rb_node.first <= rb.first && rb_node.second <= rb.second)
                return node -> sum;
            auto intersect_lb = make_pair(max(lt.first, lt_node.first), max(lt.second, lt_node.second));
            auto intersect_rb = make_pair(min(rb.first, rb_node.first), min(rb.second, rb_node.second));
            int res = 0;
            for (int i = 0; i < QTreeNode::branches; ++i) {
                res += sumQTree(node -> children[i], intersect_lb, intersect_rb);
            }

            //cout << "sum: [" << lt_node.first << " " << lt_node.second << "], [" << rb_node.first << " " << rb_node.second << "]"<< " is: " << res << endl; 
            return res;
        }

        //clean up function
        void cleanup(QTreeNode* root) {
            if (!root)  return;
            for (int i = 0; i <QTreeNode::branches; ++i)    cleanup(root->children[i]);
            delete(root);
        }
    //------------run time: 798ms, beat 2.98% ----------
};


int main() {
    vector<vector<int>> matrix({{3, 0, 1, 4, 2}, {5, 6, 3, 2, 1}, {1, 2, 0, 1, 5}, {4, 1, 0, 1, 7}, {1, 0, 3, 0, 5}});
    NumMatrix_Quad nq(matrix);
    NumMatrix n(matrix);
    cout << "correct: " << n.sumRegion(2, 1, 4, 3) << endl;
    cout << nq.sumRegion(2, 1, 4, 3) << endl;
    nq.update(3, 2, 2);

    cout << nq.sumRegion(2, 1, 4, 3) << endl;
}

