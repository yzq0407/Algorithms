//https://www.hackerrank.com/challenges/gridland-metro


#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>


using namespace std;

class Solution {
private:
    typedef pair<int, int> interval;

    int n, m, k;
    unordered_map<int, set<interval>> tracks;

public:
    Solution () {
        cin >> n >> m >> k;
        long long count = 0;
        for (int i = 0; i < k; ++i) {
            int row, start, end;
            cin >> row >> start >> end;
            auto inter = make_pair(start, end);
            if (!tracks.count(row)) {
                tracks[row].insert(inter);
                count += (end - start + 1);
            }
            else { 
                auto greater = tracks[row].upper_bound(inter);
                bool hasSmaller = greater != tracks[row].begin(), hasGreater = greater != tracks[row].end();
                if (hasSmaller) {
                    auto smaller = --greater;
                    if (smaller -> second >= start) {
                        start = smaller -> first;
                        end = max(smaller -> second, end);
                        count -= (smaller -> second - smaller -> first + 1);
                        tracks[row].erase(*smaller);
                    }
                }
                if (hasGreater) {
                    if (greater -> first <= end) {
                        end = max(greater -> second, end);
                        count -= (greater -> second - greater -> first + 1);
                        tracks[row].erase(*greater);
                    }
                }
                count += (end - start + 1);
                tracks[row].insert(make_pair(start, end));
            }
        }
        long long left = static_cast<long long>(n) * static_cast<long long>(m) - count;
        cout << left << endl;
    }
};

int main() {
    Solution s;
}
