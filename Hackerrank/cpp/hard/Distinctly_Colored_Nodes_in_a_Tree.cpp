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
        //queries.size() / 2 is the location of the answer
        queries.push_back(make_pair(make_pair(start, end), queries.size() / 2));
        queries.push_back(make_pair(make_pair(end + 1, start + n - 1), queries.size() / 2 + n));
    }

    void updateSegTree(vector<int> &seg_tree, int idx, int diff) {
        while (idx < seg_tree.size()) {
            seg_tree[idx] += diff;
            idx += idx & (-idx);
        }
    }

    int sumRange(vector<int> &seg_tree, int low, int high) {
        if (low > high) return 0;
        --low;
        int sum = 0;
        while (high) {
            sum += seg_tree[high];
            high -= high & (-high);
        }
        while (low) {
            sum -= seg_tree[low];
            low -= low & (-low);
        }
        return sum;
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
        sort(queries.begin(), queries.end(),
                [](const query& q1, const query& q2) -> bool {return q1.first.second != q2.first.second? 
                q1.first.second < q2.first.second: q1.first.first < q2.first.first;});
        query_res = vector<int>(2 * n, 0);
        vector<int> seg_tree(2 * n + 1, 0);
        //a map to keep the last occurence index of the color
        unordered_map<int, int> last_occurence;
        auto it = queries.begin();
        //i is the index in the segment tree, (i - 1) % n represents the node in the dfs_seq
        for (int i = 1; i <= 2 * n; ++i) {
            int node = dfs_seq[(i - 1) % n];
            int color = color_map[node];
            if (last_occurence.find(color) != last_occurence.end()) {
                updateSegTree(seg_tree, last_occurence[color], -1);    
            }
            last_occurence[color] = i;
            updateSegTree(seg_tree, i, 1);

            while (it != queries.end() && it -> first.second + 1== i) {
                int start = it -> first.first + 1;
                int end = it -> first.second + 1;
                int idx = it -> second;
                //cout << "interval: " << start << " " << end << endl;
                ++it;
                query_res[idx] = sumRange(seg_tree, start, end);
                //cout << "result: " << query_res[idx] << endl;
            }
        }

        long long res = 0LL;
        for(int i = 0; i <n; ++i) {
            res += 1LL * query_res[i] * query_res[i + n];
        }
        return res;
    }
};


int main() {
    Solution s;
    cout << s.solve() << endl;
}




