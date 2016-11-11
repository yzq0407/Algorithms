//https://www.hackerrank.com/challenges/maximum-subarray-sum
//


#include <iostream>
#include <vector>
#include <set>

using namespace std;

class Solution {
    private:
        typedef long long ll;
        int num_query;
        int getMaximumSubarraySum(vector<ll>& arr, ll mod) {
            ll maxSum = 0;
            ll rSum = 0;
            set<ll> sums = {0L};
            for (int num: arr) {
                rSum += num;
                rSum %= mod;
                if (rSum == mod - 1) {
                    maxSum = rSum;
                    break;
                }
                if (rSum > maxSum) {
                    maxSum = rSum;
                }
                else {
                    auto ceiling = sums.upper_bound(rSum);
                    if (ceiling != sums.end() && maxSum < rSum - *ceiling + mod) {
                        maxSum = rSum - *ceiling + mod;
                    }
                }
                sums.insert(rSum);
            }

            return maxSum;
        }

    public:
        Solution() {
            cin >> num_query;
        }

        void solve() {
            for (int i = 0; i < num_query; ++i) {
                int len;
                ll modulus;
                cin >> len >> modulus;
                vector<ll> arr(len);
                for (int j = 0; j < len; ++j) {
                    cin >> arr[j];
                }
                cout << getMaximumSubarraySum(arr, modulus) << endl;
            }
        }
};


int main() {
    Solution s;
    s.solve();

}
