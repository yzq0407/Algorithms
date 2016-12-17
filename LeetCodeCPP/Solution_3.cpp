#include <iostream>
#include <algorithm>
#include <ctime>
#include <climits>
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <cmath>
#include <set>

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
        /* 441. Arranging Coins */
    /* You have a total of n coins that you want to form in a staircase shape, where every k-th row must have exactly k coins. */

    /* Given n, find the total number of full staircase rows that can be formed. */

    /* n is a non-negative integer and fits within the range of a 32-bit signed integer. */
        int arrangeCoins(int n) {
            //n * (n + 1) / 2            n^2 / 2 + n / 2
            int low = 1;
            int high = n / 2 + 1;
            while (low <= high) {
                long long mid = static_cast<long long>(low + (high - low) / 2);
                long long total = (mid * mid + mid) / 2;
                if (total > n)  high = mid - 1;
                else    low = mid + 1;
            }
            return low - 1;
        }

/*         30. Substring with Concatenation of All Words */
/*         You are given a string, s, and a list of words, words, that are all of the same length. */
/*         Find all starting indices of substring(s) in s that is a concatenation of each word in words */ 
/*         exactly once and without any intervening characters. */

        vector<int> findSubstring(string s, vector<string>& words) {
            unordered_map<string, int> word_count;
            vector<int> res;
            if (words.size() == 0)  return res;
            int len = words[0].size();
            for (auto word: words) {
                if (word_count.find(word) == word_count.end()) {
                    word_count[word] = 1;
                }
                else {
                    ++word_count[word];
                }
            }
            for (int start = 0; start < len; ++start) {
                int i = start, j = start;
                while (i < s.size()) {
                    if (j - i == words.size() * len) {
                        res.push_back(i);
                        word_count[s.substr(i, len)] = 1;
                        i += len;
                    }
                    else if (j + len <= s.size()) {
                        string next = s.substr(j, len);
                        if (word_count.find(next) != word_count.end() && word_count[next] > 0) {
                            --word_count[next];
                            j += len;
                        }
                        else if (i != j) {
                            ++word_count[s.substr(i, len)];
                            i += len;
                        }
                        else {
                            i += len;
                            j += len;
                        }
                    }
                    else {
                        ++word_count[s.substr(i, len)];
                        i += len;
                    }

                }
            }
            return res;

        }

        /* 446. Arithmetic Slices II - Subsequence */
        /* A sequence of numbers is called arithmetic if it consists of at least three elements */ 
        /* and if the difference between any two consecutive elements is the same. */
        
        int numberOfArithmeticSlices(vector<int>& A) {
            vector<unordered_map<long, long>> subseq_count(A.size());
            int count_subseq = 0;
            for (int i = 0; i < A.size(); ++i) {
                for (int j = 0; j < i; ++j) {
                    long diff = static_cast<long>(A[i]) - static_cast<long>(A[j]);
                    if (subseq_count[i].find(diff) == subseq_count[i].end()) subseq_count[i][diff] = 0;
                    subseq_count[i][diff] += 1;
                    if (subseq_count[j].find(diff) != subseq_count[j].end()) {
                        count_subseq += subseq_count[j][diff];
                        subseq_count[i][diff] += subseq_count[j][diff];
                    }
                }
            }
            return count_subseq;
        }

        /* 452. Minimum Number of Arrows to Burst Balloons */
        int findMinArrowShots(vector<pair<int, int>> &points) {
            sort(points.begin(), points.end(), [](const pair<int, int> &p1, const pair<int, int> &p2) {
                    return p1.second != p2.second? p1.second < p2.second: p1.first < p2.first;});
            int arrow = INT_MIN, count = 0;
            for (int idx = 0; idx < points.size(); ++idx) {
                if (arrow != INT_MIN && points[idx].first <= arrow) continue;
                ++count;
                arrow = points[idx].second;
            }
            return count;
        }

        /* 456. 132 Pattern */
        /* Given a sequence of n integers a1, a2, ..., an, a 132 pattern is a subsequence ai, aj, ak such that */
        /* i < j < k and ai < ak < aj. Design an algorithm that takes a list of n numbers as input and checks */ 
        /* whether there is a 132 pattern in the list. */
        bool find132pattern(vector<int> &nums) {
            if (nums.size() < 3)    return false;
            vector<int> smaller(nums.size());
            smaller[0] = INT_MAX;
            for (int i = 1; i < nums.size(); ++i) {
                smaller[i] = min(smaller[i - 1], nums[i - 1]);
            }
            set<int> tree;
            tree.insert(nums.back());
            for (int i = nums.size() - 2; i > 0; --i) {
                auto ub = tree.lower_bound(nums[i]);
                if (ub != tree.begin()) {
                    --ub;
                    if (*ub > smaller[i])   return true; 
                }
                tree.insert(nums[i]);
            }
            return false;
        }
        
        bool isConvex(vector<vector<int>>& points) {
            if (points.size() <= 3)   return true;
            int len = points.size();
            int prev = 0;
            for (int i = 0; i < points.size(); ++i) {
                auto p0 = points[i];
                auto p1 = points[(len + i - 1) % len];
                auto p2 = points[(len + i - 2) % len];
                int x1 = p1[0] - p2[0];
                int y1 = p1[1] - p2[1];
                int x2 = p0[0] - p2[0];
                int y2 = p0[1] - p2[1];
                int curr = (x1 * y2) - (x2 * y1);
                if (curr * prev < 0)    return false;
                if (curr != 0) prev = curr;
            }
            return true;
        }
        
        
        int getMaxRepetitions(string s1, int n1, string s2, int n2) {
            const size_t len1 = s1.size(), len2 = s2.size();
            typedef pair<int, int> pair_ints;
            //technically, it is a table to record what is the last s1, s2 position combo we've seen
            //we will keep on recording such combo, if we happen to see one once again, it is a repeating pattern, we 
            //can do a shortcut calculation based on that
            vector<vector<pair_ints>> dp(len1, vector<pair_ints> (len2, make_pair(-1, -1)));
            //pos1, pos2 means the position in S1, m * S2, in the range of [0, len1 * n1), [0, m * len2 * n2) 
            int pos1 = 0, pos2 = 0;
            bool found = false;
            while (pos1 < len1 * n1) {
                if (s1[pos1 % len1] != s2[pos2 % len2]) {
                    //not match, increment pos1 only
                    ++pos1;
                }
                else {
                    if (!found && dp[pos1 % len1][pos2 % len2].first != -1) {
                        //shortcut calculation
                        auto last_pair = dp[pos1 % len1][pos2 % len2];
                        int last1 = last_pair.first, last2 = last_pair.second;
                        int diff1 = (len1 * n1 - pos1) / (pos1 - last1) * (pos1 - last1);
                        int diff2= (len1 * n1 - pos1) / (pos1 - last1) * (pos2 - last2);
                        pos1 += diff1;
                        pos2 += diff2;
                        found = true;
                    }
                    else {
                        dp[pos1 % len1][pos2 % len2] = make_pair(pos1, pos2);
                        ++pos1;
                        ++pos2;
                    }
                }
            }
            return pos2 / (len2 * n2);
        }
        
        /* 471. Encode String with Shortest Length */
        string encode(const string& s) {
            unordered_map<string, string> memo;
            return encode(s, memo);
        }

        string encode(const string& s, unordered_map<string, string>& memo) {
            if (memo.find(s) == memo.end()) {
                string multi = s;
                for (int repeat = 2; repeat <= s.size(); ++repeat) {
                    if (s.size() % repeat == 0) {
                        bool isRepeat = true;
                        int unitLen = s.size() / repeat;
                        string unit = s.substr(0, unitLen);
                        for (int start = unitLen; start < s.size(); start += unitLen) {
                            if (s.substr(start, unitLen) != unit) {
                                isRepeat = false;
                                break;
                            }
                        }
                        if (isRepeat) {
                            string compress = to_string(repeat) + "[" + encode(unit, memo) + "]";
                            if (compress.size() < multi.size()) {
                                multi = compress;
                            }
                        }
                    }
                }
                string concat = s;
                for (int breakPoint = 1; breakPoint < s.size(); ++breakPoint) {
                    string concat_str = encode(s.substr(0, breakPoint), memo)
                        + encode(s.substr(breakPoint, s.size() - breakPoint), memo);
                    if (concat_str.size() < concat.size())  concat = concat_str;
                }

                if (multi.size() < s.size() && multi.size() <= concat.size())    memo[s] = multi;
                else if (concat.size() < s.size())  memo[s] = concat;
                else    memo[s] = s;
            }
            return memo[s];
        }

};


int main() {
    vector<vector<int>> test1 = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    vector<vector<int>> test2 = {{0, 0}, {0, 10}, {10, 10}, {10, 0}, {5, 5}};
    Solution s;
    cout << s.encode("aaaaa") << endl;
    cout << s.encode("aaa") << endl;
    cout << s.encode("aabcaabcd") << endl;
    cout << s.encode("abbbabbbcabbbabbbc") << endl;
    cout << s.encode("aaaaaaaaaa") << endl;
}
