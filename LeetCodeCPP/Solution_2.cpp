#include <algorithm>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <stack>
#include <sstream>
#include <unordered_set>
#include <set>
#include <map>
#include <climits>

using namespace std;
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode (int v): val(v) {
        left = right = nullptr;
    }
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode (int v): val(v) {}
};

struct Interval {
    int start;
    int end;
    Interval(): start(0), end(0) {}
    Interval(int s, int e): start(s), end(e) {}
};

class Codec {
    public:
        int greaterStringNumber(string s1, string s2) {
            if (s1.size() != s2.size()) return s1.size() - s2.size();
            return s1 > s2? 1: s1 < s2? -1: 0;
        }


        string serialize(TreeNode *root) {
            ostringstream o;
            stack<TreeNode*> stack;
            while (root || !stack.empty()) {
                if (root) {
                    o << root -> val << " ";
                    stack.push(root);
                    root = root -> left;
                }
                else {
                    o << "#" << " ";
                    root = stack.top() -> right;
                    stack.pop();
                }
            }
            o << "#";
            return o.str();

        }

        TreeNode* deserialize(string data) {
            int pos = 0;
            return deserializeHelper(data, pos);
        }

        TreeNode* deserializeHelper(string &data, int &pos) {
            if (data[pos] == ' ')   pos++;
            if (data[pos] == '#')   {
                pos++;
                return nullptr;
            }
            int j = pos;
            while (data[pos] != ' ')  pos++;
            TreeNode* root = new TreeNode(stoi(data.substr(j, pos - j)));
            root -> left = deserializeHelper(data, pos);
            root -> right = deserializeHelper(data, pos);
            return root;
        }
};

class SmallerTreeNode {
public:
    int dup, smaller;
    long long val;
    SmallerTreeNode *left, *right;
    SmallerTreeNode(int d, int s, long long v): dup(d), smaller(s), val(v) {
        left = nullptr;
        right = nullptr;
    }
};

class Solution {
    public:
        int longestSubstring(string s, int k) {
            vector<int> after(s.size(), 0), count(26, 0);
            for (int i = s.size() - 1; i >= 0; --i) {
                after[i] = ++count[s[i] - 'a'];
            }
            fill(count.begin(), count.end(), 0);
            int start = 0, end = 0, maxLen = 0, distinct = 0, qualify = 0;
            while (end < s.size()) {
                if (count[s[end] - 'a'] + after[end] >= k) {
                    if (count[s[end] - 'a'] == k - 1)   ++qualify;
                    if (count[s[end] - 'a'] == 0)   ++distinct;
                    ++count[s[end++] - 'a'];
                    if (distinct == qualify)
                        maxLen = max(maxLen, end - start);
                }
                else {
                    while (start <= end) {
                        if (count[s[start] - 'a'] == 1) --distinct;
                        if (count[s[start] - 'a'] == k) --qualify;
                        --count[s[start++] - 'a'];
                        if (distinct == qualify)
                            maxLen = max(maxLen, end- start);
                    }
                    ++end;
                }
            }
            return maxLen;
        }

        vector<int> spiralOrder(vector<vector<int>>& matrix) {
            vector<int> res;
            if (matrix.size() == 0) return res;
            int m = matrix.size(), n = matrix[0].size();
            spiralOrder(matrix, res, 0, m - 1, 0, n - 1);
            return res;
        }

        void spiralOrder(vector<vector<int>>& matrix, vector<int>& res, int xmin, int xmax, int ymin, int ymax) {
            if (xmax < xmin || ymax < ymin) return;
            if (xmax == xmin) {
                for (int y = ymin; y <= ymax; ++y)  res.push_back(matrix[xmax][y]);
                return;
            }
            if (ymax == ymin) {
                for (int x = xmin; x <= xmax; ++x)  res.push_back(matrix[x][ymax]);
                return;
            }
            for (int y = ymin; y < ymax; ++y) {
                res.push_back(matrix[xmin][y]);
            }
            for (int x = xmin; x < xmax; ++x) {
                res.push_back(matrix[x][ymax]);
            }
            for (int y = ymax; y > ymin; --y) {
                res.push_back(matrix[xmax][y]);
            }
            for (int x = xmax; x > xmin; --x) {
                res.push_back(matrix[x][ymin]);
            }
            spiralOrder(matrix, res, xmin + 1, xmax - 1, ymin + 1, ymax - 1);
        }


        int countRangeSum(vector<int> &nums, int lower, int higher) {
            int total = 0;
            long long rangeSum = 0;
            SmallerTreeNode *root = nullptr;
            root = put(root, 0);
            for (int num: nums) {
                rangeSum += num;
                total += (searchSmaller(root, rangeSum - lower, true, 0) - searchSmaller(root, rangeSum - higher, false, 0));
                root = put(root, rangeSum); 
            }
            return total;
        }

        int searchSmaller(SmallerTreeNode *node, long long val, bool inclusive, int curr) {
            if (!node) return curr;
            if (node -> val == val){
                return curr + node -> smaller + (inclusive? node -> dup: 0);
            }
            else if (node -> val > val) {
                return searchSmaller(node -> left, val, inclusive, curr);
            }
            else {
                return searchSmaller(node -> right, val, inclusive, curr + node -> dup + node -> smaller);
            }
        }

        SmallerTreeNode* put(SmallerTreeNode *node, long long val) {
            if (!node)  return new SmallerTreeNode(1, 0, val);
            if (node -> val == val) node -> dup ++;
            else if (node -> val > val) {
                node -> smaller++;
                node -> left = put(node ->left, val);
            }
            else {
                node -> right = put(node -> right, val);
            }
            return node;
        }



        vector<int> countSmaller(vector<int> &nums) {
            SmallerTreeNode* root = nullptr;
            vector<int> res;
            for (int pos = nums.size() - 1; pos >= 0; pos--) {
                int num = nums[pos], smaller = 0;
                root = put(root, smaller, num);
                res.push_back(smaller);
            }
            return res;
        }

        SmallerTreeNode* put(SmallerTreeNode* root, int& smaller, int val) {
            if (!root)  return new SmallerTreeNode(1, 0, val);
            if (root -> val == val) {
                smaller += root -> smaller;
                return root;
            }
            if (root -> val < val) {
                smaller += (root -> smaller + root -> dup);
                return put(root -> right, smaller, val);
            }
            else {
                return put(root -> left, smaller, val);
            }
        }

        vector<vector<int>> buildingOutline(vector<vector<int>> &buildings) {
            //pair of index and in/out
            vector<pair<int, int>> crit_pts;
            for (int i = 0; i < buildings.size(); ++i) {
                crit_pts.push_back(make_pair(i, 0));
                crit_pts.push_back(make_pair(i, 1));
            }
            sort(crit_pts.begin(), crit_pts.end(), [&buildings] (const pair<int, int> &p1,
                        const pair<int, int> &p2) -> bool {
                    int x1 = buildings[p1.first][p1.second];
                    int x2 = buildings[p2.first][p2.second];
                    if (x1 != x2)   return x1 < x2;
                    if (p1.second != p2.second) return p1.second < p2.second;
                    if (p1.second == 0) return buildings[p1.first][2] > buildings[p2.first][2];
                    return buildings[p1.first][2] < buildings[p2.first][2];
                    });
            vector<vector<int>> res;
            int left = 0, height = 0;
            auto comparator = [&buildings] (const int i1, const int i2) {return buildings[i1][2] >
                buildings[i2][2];};
            multiset<int, decltype(comparator)> set(comparator);
            for (auto cp: crit_pts) {
                cout << "critical points " << buildings[cp.first][cp.second] << (cp.second == 0? " in": " out ") << endl;
                int idx = cp.first;
                //enter
                if (cp.second == 0) {
                    if (set.size() == 0 || buildings[*set.begin()][2] < buildings[idx][2]) {
                        if (set.size() == 0) {
                            left = buildings[idx][0];
                        }
                        else {
                            res.push_back(vector<int>({left, buildings[idx][0], buildings[*set.begin()][2]}));
                            left = buildings[idx][0];
                        }
                    }
                    set.insert(idx);
                }
                //exit
                else {
                    set.erase(idx);
                    int currentHigh = set.size() == 0? 0: buildings[*set.begin()][2];
                    if (buildings[idx][2] > currentHigh) {
                        res.push_back(vector<int>({left, buildings[idx][1], buildings[idx][2]}));
                        left = buildings[idx][1];
                    }
                }
            }
            return res;
            
        }


        int ladderLength(string beginWord, string endWord, unordered_set<string> &wordList) {
            unordered_set<string> set1({beginWord}), set2({endWord});
            if (beginWord == endWord)   return 0;
            int step = 1;
            return bidirectional_bfs(set1, set2, wordList, step);
        }
        
        int bidirectional_bfs(unordered_set<string> &set1, unordered_set<string> &set2, 
                unordered_set<string> &remainList, int &step) {
            ++step;
            if (remainList.size() == 0) return 0;
            unordered_set<string> next;
            for (auto it = set1.begin(), end = set1.end(); it != end; ++it) {
                string current = *it;
                cout << current << endl;
                for (int pos = 0; pos < current.size(); ++pos) {
                    char temp = current[pos];
                    for (char c = 'a'; c < 'z'; ++c) {
                        if (c == temp)  continue;
                        current[pos]  = c;
                        if (remainList.find(current) != remainList.end() && set1.find(current) == 
                                set1.end()) {
                            next.insert(current);
                            if (set2.find(current) != set2.end())   return step;
                        }
                    }
                    current[pos] = temp;
                }
                remainList.erase(*it);
            }
            if (next.size() == 0)   return 0;
            if (next.size() > set2.size())  return bidirectional_bfs(set2, next, remainList, step);
            else return bidirectional_bfs(next, set2, remainList, step);
        }


        ListNode* mergeKLists(vector<ListNode*>& lists) {
            auto comparator = [] (const ListNode* a, const ListNode* b) {return a->val < b->val;};
            priority_queue<ListNode*, vector<ListNode*>, decltype(comparator)> pq(comparator);
            ListNode* dummy(0), *p = dummy;
            for (auto list: lists) {
                if (list)   pq.push(list);
            }
            while (!pq.empty()) {
                auto node = pq.top();
                pq.pop();
                p -> next = node;
                p = p -> next;
                if (node -> next)   pq.push(node -> next);
            }
            return dummy -> next;
        }

        bool validUtf8(const vector<int>& data) {
            return validUtf8(data, 0);
        }

        

        bool validUtf8(const vector<int>& data, const int i) {
            if (i == data.size())   return true;
            //process the header
            int trailingZeros = 0;
            for (int pos = 7; pos >= 0; pos--, trailingZeros++) {
                if ((data[i] & (1 << pos)) == 0)  break;
            }
            cout << trailingZeros << " trailing" << endl;
            if (trailingZeros == 0) return validUtf8(data, i + 1);
            if (trailingZeros > 4 || trailingZeros == 1)  return false;
            for (int offset = 1; offset < trailingZeros; ++offset) {
                cout << data[i + offset] << endl;
                if (data.size() <= i + offset)  return false;
                cout << "the prefix: " << (data[i + offset] >> 6) << endl;
                if ((data[i + offset] >> 6) != 2)   return false;
            }
            return validUtf8(data, i + trailingZeros);
        }


        bool isRectangleCover(vector<vector<int>> &rectangles) {
            int area = 0, min_x = INT_MAX, max_x = INT_MIN, min_y = INT_MAX, max_y = INT_MIN;
            map<pair<int, int>, vector<pair<int, int>>> map;
            for (auto rect: rectangles) {
                pair<int, int> xin = make_pair(rect[0], 1);
                pair<int, int> xout = make_pair(rect[2], 0);
                min_x = min(rect[0], min_x);
                max_x = max(rect[2], max_x);
                min_y = min(rect[1], min_y);
                max_y = max(rect[3], max_y);
                area += (rect[2] - rect[0]) * (rect[3] - rect[1]);
                map[xin].push_back(make_pair(rect[1], rect[3]));
                map[xout].push_back(make_pair(rect[1], rect[3]));
            }
            if (area != (max_x - min_x) * (max_y - min_y))  return false;
            set<pair<int, int>> sweepline;
            for (auto it = map.begin(), end = map.end(); it != end; ++it) {
                auto cp = it -> first;
                auto intervals = it -> second;
                for (auto interval: intervals) {
                    if (cp.second == 0) {
                        printInterval(cp.first, interval, true);
                        sweepline.erase(interval);
                    }
                    else if (sweepline.size() == 0){
                        sweepline.insert(interval);
                        printInterval(cp.first, interval, false);
                    }
                    else {
                        //check if there is overlap
                        auto prev = sweepline.lower_bound(make_pair(interval.first, interval.first));
                        if (prev != sweepline.end() && prev -> first < interval.second) return false;
                        if (prev != sweepline.begin()) {
                            --prev;
                            if (prev -> second > interval.first)    return false;
                        }
                        sweepline.insert(interval);
                        printInterval(cp.first, interval, false);
                    }
                }
            }
            return true;

        }

        void printInterval(int x, pair<int, int> p, bool isRemove) {
            cout << (isRemove? "remove: ": "insert: ") << p.first << " " << p.second << " at " << x << endl;
        }



        vector<Interval> merge(vector<Interval> &intervals) {
            int pos = 1;
            for (int i = 1; i < intervals.size(); ++i) {
                if (intervals[i].start > intervals[pos - 1].end) {
                    intervals[pos++] = intervals[i];
                }
                else {
                    intervals[pos - 1].end = max(intervals[i].end, intervals[pos - 1].end);
                }
            }
            intervals.resize(pos);
            return intervals;
            
        }


        int calculateMinimumHP(vector<vector<int>> &dungeon) {
            int n = dungeon.size(), m = dungeon[0].size();
            vector<int> dp(n, 1);
            for (int i = n - 1; i >= 0; --i) {
                vector<int> aux(n, 0);
                //aux[m - 1] must go to dp[m - 1] by using dungeion[i][m - 1] health
                for (int j = m - 1; j >= 0; --j) {
                    int bestRoute = j == m - 1? dp[j]: min(aux[j + 1], dp[j]);
                    aux[j] = dungeon[i][j] > 0? max(bestRoute - dungeon[i][j], 1): (bestRoute + dungeon[i][j]);
                }
                dp = aux;
            }
            return dp[0];
        }


        int minDistance(string word1, string word2) {
            int n = word1.size(), m = word2.size();
            vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
            for (int i = 0; i <= n; ++i) {
                for (int j = 0; j <= m; ++j) {
                    if (!i || !j)   dp[i][j] = max(i, j);
                    else {
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1;
                        if (word1[i - 1] == word2[j - 1])   dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]);
                        else    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1);
                    }
                }
            }
            printVector(dp);
            return dp[n][m];
        }


        int shortestDistance(vector<vector<int>> &grid) {
            int count = 0, n = grid.size(), m = grid[0].size();
            int dx[] = {-1, 1, 0, 0}, dy[] = {0, 0, -1, 1};
            vector<vector<int>> visited(n, vector<int>(m, 0));
            vector<vector<int>> dist(n, vector<int>(m, 0));
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (grid[i][j] == 1) {
                        count++;
                        queue<pair<int, int>> queue;
                        queue.push(make_pair(i, j));
                        int distance = 0;
                        while (!queue.empty()) {
                            auto size = queue.size();
                            for (int pos = 0; pos < size; ++pos) {
                                auto current = queue.front();
                                int c_x = current.first, c_y = current.second;
                                if (grid[c_x][c_y] == 0) {
                                    dist[c_x][c_y] += distance;
                                }
                                for (int d = 0; d < 4; ++d) {
                                    int nx = c_x + dx[d], ny = c_y + dy[d];
                                    if (nx >= 0 && nx < n && ny >= 0 && ny < m && grid[nx][ny] == 0 && visited[nx][ny] < count) {
                                        if (visited[nx][ny] != count - 1)   return -1;
                                        visited[nx][ny]++;
                                        queue.push(make_pair(nx, ny));
                                    }
                                }
                                queue.pop();
                            }
                            distance++;
                        }
                        /* printVector(visited); */
                    }
                }
            }
            int max = INT_MAX;
            for (int i = 0; i < n; i++) {
                for(int j = 0; j < m; ++j) {
                    if (grid[i][j] == 0 && visited[i][j] == count) {
                        max = min(max, dist[i][j]);
                    }
                }
            }
            return max == INT_MAX? -1 : max;
        }

        void printVector(vector<vector<int>> &v) {
            for (auto it = v.begin(); it != v.end(); ++it) {
                for (auto it2 = it -> begin(); it2 != it -> end(); ++it2) {
                    cout << *it2 << " ";
                }
                cout << endl;
            }
        }

        vector<vector<int>> threeSum(vector<int>& nums) {
            unordered_map<int, int> counts;
            vector<vector<int>> res;
            if (nums.size() < 3)    return res;
            for (int num: nums) {
                counts[num]++;
            }
            for (auto it = counts.begin(), end = counts.end(); it != end; ++it) {
                int first = it -> first;
                it -> second--;
                unordered_map<int, int> temp;
                for (auto it2 = it; it2 != end; ++it2) {
                    temp[it -> first] = it -> second;
                    if (it2 -> second == 0) continue;
                    it2 -> second--;
                    int counter = 0 - first - it2 -> first;
                    if (counts.find(counter) != end && counts[counter] != 0) {
                        res.push_back(vector<int>({first, it2 -> first, counter}));
                    }
                    it2 -> second = 0;
                } 
                for (auto it2 = it; it2 != end; ++it2) {
                    counts[it2 -> first] = temp[it2 -> first];
                }
                it -> second = 0;
            }
            return res;
        }


        bool wordPatternMatch(string pattern, string str) {
            unordered_map<string, char> str_to_char;
            unordered_map<char, string> char_to_str;
            return dfs(pattern, str, str_to_char, char_to_str, 0, 0);
        }

        bool dfs(const string &pattern, const string &str, unordered_map<string, char> &str_to_char,
                unordered_map<char, string> &char_to_str, int pos_p, int pos_str) {
            if (pos_str == str.size() || pos_p == pattern.size())   
                return pos_str == str.size() && pos_p == pattern.size(); 
            for (int i = pos_str + 1; i <= str.size(); ++i) {
                string patch = str.substr(pos_str, i - pos_str);
                //if it follows the pattern
                if (char_to_str.find(pattern[pos_p]) == char_to_str.end()
                        && str_to_char.find(patch) == str_to_char.end()) {
                    str_to_char[patch] = pattern[pos_p];
                    char_to_str[pattern[pos_p]] = patch;
                    if (dfs(pattern, str, str_to_char, char_to_str, pos_p + 1, i))  return true;
                    str_to_char.erase(patch);
                    char_to_str.erase(pattern[pos_p]);
                }
                else if (char_to_str.find(pattern[pos_p]) != char_to_str.end()
                        && str_to_char.find(patch) != str_to_char.end()) {
                    if (char_to_str[pattern[pos_p]] == patch && 
                            str_to_char[patch] == pattern[pos_p] && 
                            dfs(pattern, str, str_to_char, char_to_str, pos_p + 1, i)) return true;
                }
            }
            return false;
        }
        vector<int> findSubstring(string s, vector<string> &words) {
            vector<int> res;
            if (words.size() == 0 || words[0].size() == 0)  return res;
            int len = words[0].size();
            unordered_map<string, int> map;
            for (auto it = words.begin(), end = words.end(); it != end; ++it) {
                map[*it] = map.find(*it) == map.end()? 1: map[*it] + 1;
            }
            for (int st = 0; st < len; ++st) {
                searchWords(s, res, map, words.size(), st, len);
            }
            return res;
        }

        void searchWords(string &s, vector<int> &res, unordered_map<string, int> &set, int total, int st, int len) {
            int j = st + len;
            while (j <= s.size()) {
                string sub = s.substr(j - len, len);
                if (set.find(sub) != set.end()) {
                    if (set[sub] > 0) {
                        set[sub]--;
                    }
                    else {
                        while (s.substr(st, len) != sub) {
                            set[s.substr(st, len)]++;
                            st += len;
                        }
                        st += len;
                    }
                }
                else {
                    while (st != j - len) {
                        set[s.substr(st, len)]++;
                        st += len;
                    }
                    st = j;
                }
                if (j - st == len * total)   res.push_back(st);
                j += len;
            }
        }

        vector<string> wordBreak(string s, unordered_set<string> &wordDict) {
            vector<string> res, list;
            //dp stores whether a word is breakable
            vector<bool> breakable (s.size() + 1, false);
            breakable[s.size()] = true;
            for (int i = s.size() - 1; i >= 0; --i) {
                for (int j = s.size(); j > i; j--) {
                    if (breakable[j] && wordDict.find(s.substr(i, j - i)) != wordDict.end()) {
                        breakable[i] = true;
                        continue;
                    }
                }
            }
            dfs(breakable, s, wordDict, 0, list, res);
            return res;
        }

        void dfs(const vector<bool> &breakable, const string s, 
                const unordered_set<string> &wordDict, int pos, 
                vector<string>& list, vector<string>& res) {
            if (pos == s.size()) {
                ostringstream os;
                for (auto it = list.begin(), end = list.end(); it != end; ++it) {
                    os << *it;
                    if (it + 1 != end)  os << " ";
                }
                res.push_back(os.str());
                return;
            }
            if (!breakable[pos])    return;
            for (int next = pos + 1; next <= s.size(); ++next) {
                auto frag = s.substr(pos, next - pos);
                if (wordDict.find(frag) != wordDict.end()) {
                    list.push_back(frag);
                    dfs(breakable, s, wordDict, next, list, res);
                    list.pop_back();
                }
            }
        }

        unordered_set<int> stoneSet;

        set<pair<int, int>> visited;
        bool canCross(vector<int> &stones) {
            for (int stone: stones) {
                stoneSet.insert(stone);
            }
            int first = 0; int last = stones.back();
            return dfs(first, 1, last);
        }

        bool dfs(int current, int step, int last) {
            if (current == last)    return true;
            if (stoneSet.find(current + step) == stoneSet.end() ||
                    visited.find(make_pair(current + step, step)) != visited.end())   return false;
            visited.insert(make_pair(current + step, step));
            if (step - 1 != 0 && dfs(current + step, step - 1, last))   return true;
            if (dfs(current + step, step, last) || dfs(current + step, step + 1, last))    return true;
            return false;
        }

        /* unordered_map<string, unordered_set<string>> map; */
        /* vector<vector<string>> findLadders(string beginWord, string endWord, unordered_set<string>& wordList) { */
        /*     wordList.insert(beginWord); */
        /*     wordList.insert(endWord); */
        /*     queue<string> q; */
        /*     q.push(beginWord); */
        /*     int dist = 0; */
        /*     bool found = false; */
        /*     while (!q.empty()) { */
        /*         size_t size = q.size(); */
        /*         for (size_t i = 0; i < size; ++i) { */
        /*             string current = q.front(); */
        /*             q.pop(); */
        /*             string neighbor = current; */
        /*             for (int pos = 0; pos < neighbor.size(); ++pos) { */
        /*                 char temp = neighbor[pos]; */
        /*                 for (char c = 'a'; c <= 'z'; ++c) { */
        /*                     if (c == temp)  continue; */
        /*                     neighbor[pos] = c; */
        /*                     if (wordList.find(neighbor) != wordList.end() && map.find(neighbor) == map.end()) { */
        /*                         map[current].insert(neighbor); */
        /*                         q.push(neighbor); */
        /*                         if (neighbor == endWord)    found = true; */
        /*                     } */
        /*                 } */
        /*                 neighbor[pos] = temp; */
        /*             } */
        /*         } */
        /*         dist++; */
        /*         if (found)  break; */
        /*     } */
        /*     vector<vector<string>> rs; */
        /*     vector<string> list; */
        /*     dfs(rs, beginWord, endWord, list, 0, dist); */
        /*     return rs; */
        /* } */

        /* void dfs(vector<vector<string>> &rs, const string &current, */
        /*         const string &endWord, vector<string> &list, int step, const int maxDist) { */
        /*     cout << current << endl; */
        /*     if (step > maxDist) return; */
        /*     if (current == endWord) { */
        /*         list.push_back(endWord); */
        /*         rs.push_back(list); */
        /*         list.pop_back(); */
        /*         return; */
        /*     } */
        /*     if (map.find(current) == map.end()) return; */
        /*     list.push_back(current); */
        /*     for (auto neighbor: map[current]) { */
        /*         dfs(rs, neighbor, endWord, list, step + 1, maxDist); */
        /*     } */
        /*     list.pop_back(); */
        /* } */
};

struct DLNode {
    DLNode *prev;
    DLNode *next;
    int val, key;
    DLNode(int k, int v): key(k), val(v) {prev = next = NULL;}
};

class LRUCache {
    public:
        DLNode* first;
        DLNode* last;
        const int cap;
        int current = 0;
        unordered_map<int, DLNode*> map;

        LRUCache(int capacity): cap(capacity){
            first = new DLNode(0, 0);
            last = new DLNode(0, 0);
            first -> next = last;
            last -> prev = first;
        }

        int get(int key) {
            if (map.find(key) == map.end()) return -1;
            auto rv = map[key];
            rv -> prev -> next= rv -> next;
            rv -> next -> prev= rv -> prev;
            rv -> next = first -> next;
            rv -> prev = first;
            rv -> next -> prev = rv;
            rv -> prev -> next = rv;
            return rv -> val;
        }

        void set(int key, int value) {
            if (map.find(key) != map.end()) {
                auto node = map[key];
                node -> val = value;
                node -> prev -> next = node -> next;
                node -> next -> prev = node -> prev;
                node -> next = first -> next;
                node -> prev = first;
                node -> next -> prev = node;
                node -> prev -> next = node;
            }
            else {
                auto node = new DLNode(key, value);
                current++;
                node -> prev = first;
                node -> next = first -> next;
                node -> next -> prev = node;
                node -> prev -> next = node;
                map[key] = node;
                if (current > cap) {
                    auto remove = last -> prev;
                    remove -> prev -> next = remove -> next;
                    remove -> next -> prev = remove -> prev;
                    current--;
                    map.erase(remove -> key);
                    delete (remove);
                }
            }
        }

        ~LRUCache () {
            delete(first);
            delete(last);
        }
};

int main () {
    vector<int> v({-2, 5, -1});
    Solution s;
    //SmallerTreeNode *root = new SmallerTreeNode(0, 1, 0);
    //root = s.put(root, -2);
    //root = s.put(root, 3);
    //root = s.put(root, -1);
    cout << s.countRangeSum(v, -2, 2) << endl;
}
