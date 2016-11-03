/* https://www.hackerrank.com/contests/moodysuniversityhackathon/challenges/distinctly-colored-nodes-in-a-tree */
#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>

using namespace std;

class Solution {
private:
    typedef pair<int, int> interval;
    typedef pair<interval, int> query;

    int n;
    //tree representation
    unordered_map <int, vector<int>> tree;
    //all the queries
    vector<query> queries;
    //dfs sequence
    vector<int> dfs_seq;
    //query result
    vector<int> query_res;
    
    //color    node -> color value
    vector<int> color_map;

    

    //get all the queries and its counter part
    void dfs(int node, int parent) {
        int start = dfs_seq.size();
        dfs_seq.push_back(node);
        for (int neighbor: tree[node]) {
            if (neighbor != parent) {
                dfs(neighbor, node);
            }
        }
        int end = dfs_seq.size() - 1;
        //start --- end is a query, its counter part is going to be end + 1 -----> n + start - 1
        queries.push_back(make_pair(make_pair(start, end), 1));
        queries.push_back(make_pair(make_pair(end + 1, start + n - 1), -1));
    }

public:
    Solution() {
        //create trees
        cin >> n;
        color_map.reserve(n + 1);
        for (int i = 1; i <= n; ++i) {
            cin >> color_map[i];
        }
        for (int i = 0; i < n - 1; ++i) {
            int one, other;
            cin >> one;
            cin >> other;
            tree[one].push_back(other);
            tree[other].push_back(one);
        }
    }

    long long solve() {
        dfs(1, -1);
        return -1L;
    }
};


int main() {
    Solution s;
    cout << s.solve() << endl;
}




