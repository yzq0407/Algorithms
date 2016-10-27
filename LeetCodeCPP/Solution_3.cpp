#include <iostream>
#include <algorithm>
#include <ctime>
#include <climits>
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <cmath>

using namespace std;
class Solution {
    public:
        long long getNumberOfFills(int dimension) {
            unordered_map <int, long long> dp({{0, 1}});
            for (int total = 1; total <= dimension * dimension; ++total) {
                vector<int> array(dimension, 0);
                fillTable(0, array, total, dp, 0);
            }
            vector<int> final(dimension, dimension);
            return dp[hash(final)];
        }

        void fillTable(int pos, vector<int>& array, int total, unordered_map<int, long long>& dp, int prevTotal) {
            if (pos == array.size()) {
                long long sum = 0;
                for (int i = 0; i < array.size(); ++i) {
                    if (array[i] == 0)  break;
                    if (i == array.size() - 1 || array[i] > array[i + 1]) {
                        --array[i];
                        sum += dp[hash(array)];
                        ++array[i];
                    }
                }
                dp.insert({hash(array), sum});
            }
            else {
                for (int possible = (int) ceil(((double) total - (double) prevTotal) / (array.size() - pos));
                        possible <= min(pos == 0? INT_MAX: array[pos - 1], min((int)array.size(), total - prevTotal)); ++possible) {
                    array[pos] = possible;
                    fillTable(pos + 1, array, total, dp, prevTotal + possible);
                }
            }
        }

        int hash(vector<int> &array) {
            int res = 0;
            for (int num: array) {
                res = (res << 3) ^ num;
            }
            return res;
        }

        vector<string> findStrobogrammatic(int n) {
            if (n == 1) return {"1", "8", "0"};
            vector<string> res;
            queue<string> q({"", "1", "8", "0"});
            while (!q.empty()) {
                string core = q.front();
                q.pop();
                if (core.size() == n && core[0] != '0') res.push_back(core);
                if (core.size() >= n - 2) {
                    q.push("0" + core + "0");
                    q.push("1" + core + "1");
                    q.push("6" + core + "9");
                    q.push("9" + core + "6");
                    q.push("8" + core + "8");
                }
            }
            return res;
        }
        int totalNQueen(int n) {
            int total = 0;
            dfs(0, 0, 0, 0, n, total);
            return total;
        }

        void dfs(int row, int col_bit, int diag_bit, int adiag_bit,
                int n, int &total) {
            int pos = ((1 << n) - 1) & ~(col_bit | (diag_bit >> row) |
                    (adiag_bit  >> (n - 1 - row)));
            while (pos != 0) {
                int lowbit = pos & (-pos);
                pos ^= lowbit;
                if (row == n - 1)   ++total;
                else    dfs(row + 1, col_bit ^ lowbit, diag_bit ^ (lowbit << row), 
                        adiag_bit ^ (lowbit << (n - 1 - row)), n, total);
            }
        }

        int characterReplacement(string s, int k) {
            vector<int> count(26, 0);
            int i = 0, j = 0, maxCount = 0;
            while (i < s.size()) {
                maxCount = max(maxCount, ++count[s[i++] - 'A']);
                int repl = i - j - maxCount;
                if (repl > k)   --count[s[j++]];
            }
            return i - j;
        }

        /* 439. Ternary Expression Parser */
        /* Given a string representing arbitrarily nested ternary expressions, calculate the result of the expression. */
        /* You can always assume that the given expression is valid and only consists of digits 0-9, ?, :, T and F (T and F represent True and False respectively). */
        /* string parseTernary(string expression) { */
        /*     int pos = 0; */
        /*     return parseTernary(expression, pos); */
        /* } */

        /* char parseTernary(const string& expression, int& pos) { */
        /*     if (expression[pos] != 'T' && expression [pos] != 'F') { */
        /*         return expression[pos++]; */
        /*     } */
        /*     bool eval = expression[pos++] == 'T'; */
        /*     string left = parseTernary(expression, ++pos); */
        /*     string right = parseTernary(expression, ++pos); */
        /*     return eval?left: right; */
        /* } */



};


int main() {
    Solution s;
    cout << s.parseTernary("F?2:3") << endl;
}
