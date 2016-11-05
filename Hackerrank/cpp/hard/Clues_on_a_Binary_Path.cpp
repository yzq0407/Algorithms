/* Clues on a Binary Path */
/* https://www.hackerrank.com/contests/hourrank-14/challenges/clues-on-a-binary-path */

#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

using namespace std;


class Solution {
private:
    struct TrieNode {
        unordered_set<int> reached;
        int depth;

        TrieNode* zero_child;
        TrieNode* one_child;

        TrieNode(int d): depth(d), zero_child(nullptr), one_child(nullptr) {
        }

        ~TrieNode() {
            if (zero_child) delete zero_child;
            if (one_child) delete one_child;
        }
    };


    typedef pair<int, int> road_to;
    typedef vector<pair<int, int>> adjs;
    int n, m, d;
    int count;
    unordered_map<int, adjs> tree;

    TrieNode* dfs(int city, TrieNode* node, int depth) {
        if (!node)  {
            node = new TrieNode(depth);
            if (depth == d) {
                ++count;
            }
        }
        if (depth == d) return node;
        node -> reached.insert(city);
        for (road_to road: tree[city]) {
            int neighbor = road.first;
            int mark = road.second;
            if (mark) {
                if (!node -> one_child || node -> one_child -> reached.find(neighbor) 
                        == node -> one_child -> reached.end()) {
                    node -> one_child = dfs(neighbor, node -> one_child, depth + 1);
                }
            }
            else {
                if (!node -> zero_child || node -> zero_child -> reached.find(neighbor)
                        == node -> zero_child -> reached.end()) {
                    node -> zero_child = dfs(neighbor, node -> zero_child, depth + 1);
                }
            }
        }
        return node;
    }

public:
    Solution(): count(0) {
        cin >> n >> m >> d;
        for (int i = 0; i < m; ++i) {
            int one, other, digit;
            cin >> one >> other >> digit;
            tree[one].push_back(make_pair(other, digit));
            if (other != one)
                tree[other].push_back(make_pair(one, digit));
        }
    }
    
    int solve() {
        dfs(1, nullptr, 0);
        return count;
    }
};

int main() {
    Solution s;
    cout << s.solve() << endl;

}
