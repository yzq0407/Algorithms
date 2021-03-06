#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <climits>
#include <set>
#include <queue>
#include <stack>
#include <unordered_set>
#include <ostream>
#include "accessories.h"
#include "utils.h"



using namespace std;
//contains all the solution functions for the favoriate questions
class Solution {
public:
    /* 354. Russain Doll Envelopes */
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
    //----running time 33ms, beats 89.18%------//


/* 282. Expression Add Operators */
/* Given a string that contains only digits 0-9 and a target value, */ 
/* return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value. */

    vector<string> addOperators(string num, int target) {
        //using dfs to search and see if a expression is evaluated as the given value
        //since * and / has a higher precedence than + and -, we need to record the last value we've seen and determine whether to update that
        vector<string> res;
        dfs(res, "", num, 0, 0, 0, target);
        return res;
    }

    //helper function to to the dfs, expression is passed by value to do efficient backtracking
    void dfs(vector<string> &res, string expression, const string& nums, int pos, 
            long current, long last_num, int target) {
        if (pos == nums.size() && current == target) {
            res.push_back(expression);
            return;
        }
        if (pos == nums.size()) return;
        for (int len = 1; len <= min((int)nums.size() - pos, 10); ++len) {
            if (nums[pos] == '0' && len > 1)    break;
            long next = stol(nums.substr(pos, len));
            if (next > INT_MAX) break;
            if (pos == 0) {
                dfs(res, to_string(next), nums, len, next, next, target);
                continue;
            }
            //we will need to update 
            dfs(res, expression + "+" + to_string(next), nums, pos + len, current + next, next, target);
            dfs(res, expression + "-" + to_string(next), nums, pos + len, current - next, -next, target);
            //this is the important step, pay attention to the way that current is being updated to keep in pace
            dfs(res, expression + "*" + to_string(next), nums, pos + len, current - last_num + last_num * next, last_num * next, target);
        }
    }

    /* 391. Perfect Rectangle */
    /* Given N axis-aligned rectangles where N > 0, determine if they all together form an exact cover of a */
    /* rectangular region. */

    /* Each rectangle is represented as a bottom-left point and a top-right point. For example, a unit square */
    /* is represented as [1,1,2,2]. (coordinate of bottom-left point is (1, 1) and top-right point is (2, 2)). */
    bool isRectangleCover(vector<vector<int>> &rectangles) {
        //To check if it is a exact cover of rectangle, it needs to satisfy two conditions:
        //1. The sum of the areas of all the rectangles must be equal to the big square (suppose it is a square)
        //2. There is no overlapping happening within the whole range
        //So my solution is just based on these two observations
        //crit_pts represents all the critical points in xmin or xmax corresponding to the interval [ymin, ymax]
        //the first attribute represent the index in the rectangle, the second represent in(0) or out(2)
        vector<pair<int, int>> crit_pts;
        crit_pts.reserve(rectangles.size() * 2);
        int xmin = INT_MAX, ymin = xmin, ymax = INT_MIN, xmax = ymax, totalArea = 0;
        for (int i = 0; i < rectangles.size(); ++i) {
            xmin = min(rectangles[i][0], xmin);
            ymin = min(rectangles[i][1], ymin);
            xmax = max(rectangles[i][2], xmax);
            ymax=  max(rectangles[i][3], ymax);
            totalArea += (rectangles[i][2] - rectangles[i][0]) * (rectangles[i][3] - rectangles[i][1]);
            crit_pts.push_back(make_pair(i, 0));
            crit_pts.push_back(make_pair(i, 2));
        }
        //if it does not satisfy the first condition, return false
        if (totalArea != (xmax - xmin) * (ymax - ymin)) return false;
        //sort based on the two criteria: smaller x value precede greater, when a tie happens, put out point before
        //in point
        sort(crit_pts.begin(), crit_pts.end(), [&rectangles] (const pair<int, int> &p1, const pair<int, int> &p2)
            -> bool {
                int x1 = rectangles[p1.first][p1.second], x2 = rectangles[p2.first][p2.second];
                return x1 != x2? x1 < x2: p2.second < p1.second;
            });
        //scan each point, maintain a treeset that can quickly search whether there is overlapping
        auto compare = [&rectangles](int a, int b) -> bool {
            return rectangles[a][1] < rectangles[b][1];
        };
        set<int, decltype(compare)> scan_line(compare);
        for (auto crit_pt: crit_pts) {
            if (crit_pt.second == 0) {
                //in point, need to do a set insert, first check if there is any overlap
                if (scan_line.upper_bound(crit_pt.first) != scan_line.end()) {
                    int interval_start = rectangles[*scan_line.upper_bound(crit_pt.first)][1];
                    if (interval_start < rectangles[crit_pt.first][3])  return false;
                }
                if (scan_line.upper_bound(crit_pt.first) != scan_line.begin()) {
                    int interval_end = rectangles[*--scan_line.upper_bound(crit_pt.first)][3];
                    if (interval_end > rectangles[crit_pt.first][1])    return false;
                }
                //do insert
                scan_line.insert(crit_pt.first);
            }
            else {
                //out point, do a erase operation to remove the point
                scan_line.erase(crit_pt.first);
            }
        }
        return true;
        // ------ run time 426 ms beats 22.81% -------
    }

    //we can see that the above solution is kind of slow, and it runs O(nlogn)
    //there is actually a better solution which runs in O(n), it only needs to do two things
    //1. exactly the same as we did in the first solution, assert the area is equal
    //2. all the indices must be even except the ones in four corners
    //Implementation becomes trivial
    bool isRectangleCover2 (vector<vector<int>> &rectangles) {
        using coordinates = pair<int, int>;
        auto hash = [] (const coordinates& p) {return (size_t)p.first *31 + p.second;};
        auto equal = [] (const coordinates& p1, const coordinates& p2) {
            return p1.first == p2.first && p1.second == p2.second;};
        unordered_map<coordinates, int, decltype(hash), decltype(equal)> corner_count(rectangles.size(), hash, equal);

        int xmin = INT_MAX, ymin = xmin, ymax = INT_MIN, xmax = ymax, totalArea = 0;
        for (int i = 0; i < rectangles.size(); ++i) {
            xmin = min(rectangles[i][0], xmin);
            ymin = min(rectangles[i][1], ymin);
            xmax = max(rectangles[i][2], xmax);
            ymax=  max(rectangles[i][3], ymax);
            totalArea += (rectangles[i][2] - rectangles[i][0]) * (rectangles[i][3] - rectangles[i][1]);
            //insert four corners
            for (int x = 0; x <= 2; x +=2) {
                for (int y = 1; y <= 3; y += 2) {
                    auto coord = make_pair(rectangles[i][x], rectangles[i][y]);
                    if (corner_count.find(coord) == corner_count.end()) corner_count[coord] = 1;
                    else    ++corner_count[coord];
                }
            }
        }
        if (totalArea != (xmax - xmin) * (ymax - ymin)) return false;
        if (corner_count[make_pair(xmin, ymin)] != 1 || corner_count[make_pair(xmax, ymin)] != 1 ||
                corner_count[make_pair(xmax, ymax)] != 1 || corner_count[make_pair(xmin, ymax)] != 1)
            return false;
        corner_count.erase(make_pair(xmin, ymin));
        corner_count.erase(make_pair(xmin, ymax));
        corner_count.erase(make_pair(xmax, ymax));
        corner_count.erase(make_pair(xmax, ymin));
        for (auto it = corner_count.begin(); it != corner_count.end(); ++it) {
            if (it -> second % 2)   return false;
        }
        return true;
        //-----------run time 202 ms beats 79.65% ---------------
    }
       /* 309 Best Time to Buy and Sell Stock with Cooldown */
/*     Say you have an array for which the ith element is the price of a given stock on day i. */

/*     Design an algorithm to find the maximum profit. You may complete as many transactions as you like */
/*     (ie, buy one and sell one share of the stock multiple times) with the following restrictions: */

/*     You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again). */
/*     After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day) */
    int maxProfit(vector<int>& prices) {
        //basic idea is to maintain two dp array, one is we current do not hold stock in hand(sell)
        //the other one is we hold stock currently in hand(buy), then we should have the substructure:
        //sell[i] = max(sell[i - 1], buy[i - 1] + prices[i])
        //buy[i] = max(buy[i - 1], sell[i - 2] - prices[i])
        if (prices.size() < 2) return 0;
        vector<int> sell(prices.size(), 0), buy(prices.size(), 0);
        //first day the buy entry must be -prices[0]
        buy[0] = -prices[0];
        for (int i = 1; i < prices.size(); ++i) {
            sell[i] = max(sell[i - 1], buy[i - 1] + prices[i]);
            buy[i] = max(buy[i - 1], sell[max(i - 2, 0)] - prices[i]);
        }
        return sell[prices.size() - 1];
        //--------run time 6ms beats 26.99%-----------
    }

    /* 300. Longest Increasing Subsequence */
    /* Given an unsorted array of integers, find the length of longest increasing subsequence. */
    vector<int> lengthOfLIS(vector<int>& nums) { 
        //maintain a tail array to keep all the possible length's tail
        //follow up:
        //if we want to recover the sequence, how to do it?
        //we can have an array to keep the previous pointer, each time we insert a new element, we record it's previous
        //element (in this way, we cannot keep the real value in the tail's array but their index in the original array
        vector<int> tails, predecessor(nums.size(), 0);
        for (int i = 0; i < nums.size(); i++) {
            //if it is greater than the current end of tails, we added to the end
            if (tails.empty() || nums[i] > nums[tails.back()]) {
                tails.push_back(i);
                predecessor[i] = tails.size() == 1? -1: tails[tails.size() - 2];
            }
            else {
                //binary search
                int low = 0;
                int high = tails.size() - 1;
                while (low <= high) {
                    int mid = (low + high) >> 1;
                    if (nums[tails[mid]] >= nums[i])   high = mid - 1;
                    else    low = mid + 1;
                }
                //high + 1 is the right possition
                tails[high + 1] = i;
                predecessor[tails[high + 1]] = high == -1? -1: tails[high];
            }
        }
        vector<int> sequence(tails.size(), 0);
        for (int i = tails.size() - 1, current = tails.back(); i >= 0; i--) {
            sequence[i] = nums[current];
            current = predecessor[current];
        }
        return sequence;
        // ------ run time 3 ms  beats 65.12% ---------
    }

    /* 212. Word Search II */
    /* Given a 2D board and a list of words from the dictionary, find all words in the board. */

    /* Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" */
    /* cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word. */
    vector<string> findWords(vector<vector<char>>& board, vector<string> &words) {
        //here we need to use trie tree to insert all the words such that we can efficiently do dfs
        //1. we do not need a full trie, trienode is fast enough
        //2. each node must maintain a field called word, if it is terminating node, we set it to be the string we insert
        //3. everytime we find the string, we set the word field to be empty string
        vector<string> res;
        if (board.size() == 0 || words.size() == 0) return res;
        TrieNode* root = nullptr;
        for (auto word: words) {
            root = put(root, word, 0);
        }
        vector<int> dx({-1, 1, 0, 0});
        vector<int> dy({0, 0, -1, 1});
  
        for (size_t i = 0; i < board.size(); ++i) {
            for (size_t j = 0; j < board[0].size(); ++j) {
                dfs(res, root -> children[board[i][j] - 'a'], board, i, j, dx, dy);
            }
        }
        //do the cleanup
        cleanup(root);
        return res;
        //----------run time 138ms beats 40.73%---------
    }

    TrieNode *put(TrieNode *root, const string& word, int pos) {
        if (!root)  root = new TrieNode();
        if (pos == word.size()) root -> val = word;
        else {
            root -> children[word[pos] - 'a'] = put(root -> children[word[pos] - 'a'], word, pos + 1);
        }
        return root;
    }
    
    void dfs(vector<string> &res, TrieNode *node, vector<vector<char>> &board, 
            int x, int y, const vector<int>& dx, const vector<int>& dy) {
        if (!node)  return;
        if (node -> val.size()) {
            res.push_back(node -> val);
            node -> val = "";
        }
        char temp = board[x][y];
        board[x][y] = '#';
        for (size_t d = 0; d < 4; ++d) {
            int nx = dx[d] + x;
            int ny = dy[d] + y;
            if (nx >= 0 && nx < board.size() && ny >= 0 && ny < board[0].size() && board[nx][ny] != '#') {
                dfs(res, node -> children[board[nx][ny] - 'a'], board, nx, ny, dx, dy);
            }
        }
        board[x][y] = temp;
    }

    void cleanup(TrieNode* root) {
        if (!root)  return;
        for (int i = 0; i < 26; ++i)    cleanup(root -> children[i]);
        delete(root);
    }

/*     128. Longest Consecutive Sequence */
/*     Given an unsorted array of integers, find the length of the longest consecutive elements sequence. */
    int longestConsecutive(vector<int> &nums) {
        //idea: keep a map that takes (start, end) and (end, start) as KV pairs
        //each time we see a new value, makesure it did show up before, and check is there a end or start that is one
        //number away, there could be four situations
        if (nums.size() == 0)   return 0;
        //the map must be 
        unordered_map<int, int> intervals;
        unordered_set<int> visited;
        int maxSequence = 1;
        for (int num: nums) {
            if (visited.find(num) != visited.end()) continue;
            visited.insert(num);
            int head = num, end = num;
            if (num != INT_MIN && intervals.find(num - 1) != intervals.end()) {
                head = intervals[num - 1];
                intervals.erase(intervals[num - 1]);
                intervals.erase(num - 1);
            }
            if (num != INT_MAX && intervals.find(num + 1) != intervals.end()) {
                end = intervals[num + 1];
                intervals.erase(intervals[num + 1]);
                intervals.erase(num + 1);
            }
            maxSequence = max(end - head + 2, maxSequence);
            intervals.insert({head, end});
            intervals.insert({end, head});
        }
        return maxSequence;
        //----------run time 43ms beats 11.82%-----------
    }
    

    /* 286 Walls and Gates */
    /* You are given a m x n 2D grid initialized with these three possible values. */

    /* -1 - A wall or an obstacle. */
    /* 0 - A gate. */
    /* INF - Infinity means an empty room. We use the value 231 - 1 = 2147483647 to */ 
    /* represent INF as you may assume that the distance to a gate is less than 2147483647. */
    /* Fill each empty room with the distance to its nearest gate. */ 
    /* If it is impossible to reach a gate, it should be filled with INF. */
    void wallsAndGates(vector<vector<int>>& rooms) {
        //Idea is to use bfs, but we want to search it at the same time, that is
        //inject all the gate into the queue and each time exact a layer of it
        queue<pair<int, int>> q;
        for (int i = 0; i < rooms.size(); ++i) {
            for (int j = 0; j < rooms[0].size(); ++j) {
                if (rooms[i][j] == 0) q.push(make_pair(i, j));
            }
        }
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        int dist = 1;
        while (!q.empty()) {
            size_t size = q.size();
            for (size_t i = 0; i < size; ++i) {
                auto pos = q.front();
                q.pop();
                for (int d = 0; d < 4; ++d) {
                    int nx = dx[d] + pos.first, ny = dy[d] + pos.second;
                    if (nx >= 0 && nx < rooms.size() && ny >= 0 && ny < rooms[0].size() && rooms[nx][ny] == INT_MAX) {
                        rooms[nx][ny] = dist;
                        q.push(make_pair(nx, ny));
                    }
                }
            }
            ++dist;
        }
        //-------run time 119ms   beats 76.15%---------
    }

/*     275. H-Index II */
/*     Follow up for H-Index: What if the citations array is sorted in ascending order? */ 
/*     Could you optimize your algorithm? */
    int hIndex(vector<int>& citations) {
        //we are looking for the lowest i such that citations n - i <= citations[i]
        //keep the invariant property such that citations[high] <= n - i
        if (citations.size() == 0)  return 0;
        int low = 0;
        int high = citations.size() - 1;
        while (low <= high) {
            int mid = (low + high) >> 1;
            if (citations[mid] >= citations.size() - mid)   high = mid - 1;
            else    low = mid + 1;
        }
        return citations.size() - high - 1;
        // -----run time 9ms, beats 46.72%--------
    }


/*     324. Wiggle Sort II */
/*     Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3].... */
    void wiggleSort_naive(vector<int> &nums) {
        //trivial solution, sort the nums, and then order it using another array
        vector<int> copy = nums;
        sort(copy.begin(), copy.end());
        for (int medium = nums.size() / 2, i = 0; medium >= 0; medium--, i += 2) {
            nums[i] = copy[medium];
        }
        for (int medium = nums.size() - 1, i = 1; medium > nums.size() / 2 && i < nums.size(); medium--, i += 2) {
            nums[i] = copy[medium];
        }
        //------run time 136ms-- beats 21.76%-------
    }

    void wiggleSort_refined(vector<int> &nums) {
        //In order to solve it in O(1) space, we need a virtual indexing technique
        //eventually we want our final array to be like this:
        //M   M   S   S
        //  L   L   M
        //where M means median, S means small and L means large
        //so in a sorted array, we want to map their original index into their final index
        //odd length case
        //prev    0    1    2    3    4    5    6
        //after   6    4    2    0    5    3    1
        //even length case
        //prev    0    1    2    3    4    5
        //after   4    2    0    5    3    1
        //the function is (2 * (len - i - 1) + 1) % (n | 1)
        //note that this function works for both odd and even number length
        sort(nums.begin(), nums.end());
        //now what we need to do is to keep on doing swapping, since it is a closed cycle
        //we only need to do len - 1 times
        long long flag = (1 << nums.size()) - 1;
        while (flag) {
            int mask = (flag & -flag);
            flag ^= mask;
            cout << mask << endl;
            int idx = log2(mask);
            int current = idx;
            cout << idx << endl;
            do {
                int to = virtual_index(current, nums.size());
                swap(nums, idx, current);
                current = to;
                flag &= ~(1 << current);
            } while (current != idx);
        }
        //-----this will fail the large case since it will exceeds 64 bits-----//
    }

    int virtual_index(int idx, int len) {
        //this function implement the above index transformation
        cout << "idx: " << idx << " mapped to " << (2 * (len - idx - 1) + 1) % (len | 1) << endl;
        return (2 * (len - idx - 1) + 1) % (len | 1);
    }

    void swap(vector<int> &nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    void wiggleSort_perfect(vector<int> &nums) {
        //note that we do not need to know the whole sequence of array, we only need to know the median
        //specifically, the (len / 2 - 1) element in the sorted position
        //after we find this element, we can simply say that the element < median will be partitioned to the 
        //small part(keep a virtual index pointer) and any element > median will be partitioned to the 
        //large part(keep a virtual index pointer)
        //---------------------------------------------------------------------------------------------
        //quick partition
        if (nums.size() < 2)    return;
        quickSelect(nums, 0, nums.size() - 1, nums.size() / 2);
        //this is not strictly the medium (not for even number length array)
        int medium = nums[nums.size() / 2];
        int small = 0, large = nums.size() - 1, pos = 0;
        for (int i = 0; i < nums.size(); ++i) {
            //traverse the whole array by the virtual index order
            if (nums[virtual_index(pos, nums.size())] < medium) 
                swap(nums, virtual_index(pos++, nums.size()), virtual_index(small++, nums.size()));
            else if (nums[virtual_index(pos, nums.size())] > medium) 
                swap(nums, virtual_index(large--, nums.size()), virtual_index(pos, nums.size()));
            else ++pos;
        }

    }

    void quickSelect(vector<int> &nums, int start, int end, int pos) {
        //this function implement the quickSelect algorithm, it has amortized O(n) time
        //to get guaranteed O(n) time, we need to do a "median of medians" partition, which takes much more effort to implement
        //get a random pivot
        if (start >= end)   return;
        int pivot = rand() % (end - start + 1) + start;
        int pivot_val = nums[pivot];
        swap(nums, pivot, start);
        int small = start, large = end, i = start + 1;
        while (small < large) {
            if (nums[i] <= pivot_val)   nums[small++] = nums[i++];
            else    swap(nums, i, large--);
        }
        nums[small] = pivot_val;
        if (small == pos)   return;
        if (small < pos)    quickSelect(nums, small + 1, end, pos);
        else    quickSelect(nums, start, small - 1, pos);
        //use this we can easy solution to the 215 Kth Largest Element in an Array which beats 85.17%
    }
    
    /* 45. Jump Game II */
    /* Given an array of non-negative integers, you are initially positioned at the first index of the array. */

    /* Each element in the array represents your maximum jump length at that position. */

    /* Your goal is to reach the last index in the minimum number of jumps. */
    int jump(vector<int> &nums) {
        //greedy approach, record the current farthest and next farthest
        //if it goes beyond the current farthest, trigger a jump to next farthest
        int step = 0, limit = 0, next = 0;
        for (int i = 0; i < nums.size(); ++i) {
            next = max(i + nums[i], next);
            if (i == limit) {
                limit = next;
                ++step;
            }
            if (limit >= nums.size())   break;
        }
        return step;
        //-------run time 16ms------beats 26.64%------------
    }

    /* 33. Search in Rotated Sorted Array */
    /* Suppose a sorted array is rotated at some pivot unknown to you beforehand. */

    /* (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2). */

    /* You are given a target value to search. If found in the array return its index, otherwise return -1. */

    /* You may assume no duplicate exists in the array. */
    int search(vector<int>& nums, int target) {
        //when we choose a mid, we have to determine whether it is in the left half 
        //of rotated array or right half of the rotated array
        //We should have four different scenario based on this approach
        int low = 0;
        int high = nums.size() - 1;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (nums[mid] >= nums[low]) {
                if (target >= nums[low] && target <= nums[mid]) {
                    high = mid;
                }
                else {
                    low = mid + 1;
                }
            }
            else {
                if (target > nums[mid] && target <= nums[high]) {
                    low = mid + 1;
                }
                else high = mid;
            }
        }
        return nums[low] == target? low: -1;
        //------run time 9ms------beats 1.32%
    }

    /* 10. Regular Expression Matching */
    /* Implement regular expression matching with support for '.' and '*'. */
    bool isMatch(string s, string p) {
        //use dynamic programming to solve this problem, let dp[i][j] be 
        //whether s[0:i] and p[0:j] matched or not, so we should have the following property:
        //dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.') (last character matched)
        //or dp[i][j] = (dp[i - 1][j] &&(s[i - 1] == p[j - 2] ||p[j - 2] =='.') || dp[i][j - 2] if (p[j - 1] == '*')
        bool dp[s.size() + 1][p.size() + 1];
        // zero strings always matched
        dp[0][0] = true;
        for (int i = 1; i <= s.size(); ++i) dp[i][0] = false;
        for (int j = 1; j <= p.size(); ++j) {
            dp[0][j] = p[j - 1] == '*'? dp[0][j - 2]: false;
        }
        for (int i = 1; i <= s.size(); ++i) {
            for (int j = 1; j <= p.size(); ++j) {
                dp[i][j] = false;
                if (s[i - 1] == p[j - 1] || p[j - 1] == '.')    dp[i][j] = dp[i - 1][j - 1];
                if (p[j - 1] == '*')    dp[i][j] = dp[i][j] || dp[i][j - 2] 
                    || (dp[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.'));
                //cout << i << " " << j << " " << dp[i][j];
            }
        }
        return dp[s.size()][p.size()];
    }

    /* 407. Trapping Rain Water II */
    /* Given an m x n matrix of positive integers representing the height of */
    /* each unit cell in a 2D elevation map, compute the volume of water it is able */
    /* to trap after raining. */
    int trapRainWater(vector<vector<int>>& heightMap) {
        //we want to start from all the boundaries to seach into inside, since for each node
        //the water it can keep is actually:
        //1. find the shortest path to the boundary
        //2. Use the highest point in the shortest path to compare with that node
        //3. If it is larger than the node, we know the water reservation is gonna be 0
        //We donnot need to calculate shortest path for every node. we can do the other way
        //around, just push triplet(x, y, height) into pq, everytime pop a triplet with lowest height
        //out of pq, search all its neighbors, if any of the neighbor has height < triplet.height,
        //add the difference into the pq. Finally add all the neighbor into pq
        if (heightMap.size() == 0)  return 0;
        using triplet = pair<int, pair<int, int>>;
        //this is a min pq, reverse it using lambda
        auto compare = [] (const triplet& t1, const triplet& t2) {return t1.first > t2.first;};
        priority_queue<triplet, vector<triplet>, decltype(compare)> pq(compare);
        int sum = 0, n = heightMap.size(), m = heightMap[0].size();
        for (int i = 0; i < n; ++i) {
            pq.push(make_pair(heightMap[i][0], make_pair(i, 0)));
            pq.push(make_pair(heightMap[i][m - 1], make_pair(i, m - 1)));
            heightMap[i][0] = heightMap[i][m - 1] = -1;
        }
        for (int j = 1; j < m - 1; ++j) {
            pq.push(make_pair(heightMap[0][j], make_pair(0, j)));
            pq.push(make_pair(heightMap[n - 1][j], make_pair(n - 1, j)));
            heightMap[0][j] = heightMap[n - 1][j] = -1;
        }
        const int dx[] = {-1, 1, 0, 0};
        const int dy[] = {0, 0, -1, 1};
        while (!pq.empty()) {
            auto trip = pq.top();
            pq.pop();
            int x = trip.second.first, y = trip.second.second;
            for (int d = 0; d < 4; ++d) {
                int nx = dx[d] + x;
                int ny = dy[d] + y;
                if (nx >= 0 && nx < n && ny >= 0 && ny < m && heightMap[nx][ny] != -1) {
                    pq.push(make_pair(max(heightMap[nx][ny], trip.first), make_pair(nx, ny)));
                    if (heightMap[nx][ny] < trip.first) sum += trip.first - heightMap[nx][ny];
                    heightMap[nx][ny] = -1;
                }
            }
        }
        return sum;
    }

    /* 47 Permutations II */
    /* Given a collection of numbers that might contain duplicates, return all possible unique permutations. */
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        //since there are duplicates, every time we do swap, we have to make sure there is no same number 
        //before itself
        vector<vector<int>> res;
        dfs(res, nums, 0);
        return res;
        //----run time 23ms, beats 92.92%----- 
    }

    void dfs(vector<vector<int>> &res, vector<int>& nums, int idx) {
        if (idx == nums.size()) {
            res.push_back(nums);
            return;
        }
        for (int i = idx; i < nums.size(); ++i) {
            if (containsDuplicateBefore(nums, idx, i))   continue;
            swap(nums, idx, i);
            dfs(res, nums, idx + 1);
            swap(nums, idx, i);
        }
    }

    bool containsDuplicateBefore(vector<int>& nums, int from, int idx) {
        for (int i = from; i < idx; ++i) {
            if (nums[idx] == nums[i])   return true;
        }
        return false;
    }

    /* 37 Sudoku Solver */
    /* Write a program to solve a Sudoku puzzle by filling the empty cells. */
    bool solveSudoku(vector<vector<char>> &board) {
        //Each time we try to fill an empty cell, need to check if there is an cell
        //in the same row, column and block that has the same value. 
        //To elimnate checking 24 cells each time, instead we keep a bitmask for each cell
        //, row and block we represent which number is being occupied in this cell/row/block
        //if number n is occupied, then we set the (n - 1) digit in the bitmask to be 1
        //Thus, we need 9 row_mask, 9 col_mask and 9 block_mask, each time we are filling a
        //cell, just do a bit & on all the inverse of the corresponding mask and we can get
        //all the "available possitions", using low_bit function to retrieve all the numbers
        //that we can fit(log2 function is implemented iteratively, don't use that)
        //-----------------Preprocess------------------------------
        vector<int> row_mask(9, 0), col_mask(9, 0), block_mask(9, 0);
        unordered_map<int, char> pos_to_char;
        for (int i = 0; i < 9; ++i) pos_to_char[1 << i] = (char) ('1' + i);
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] != '.') {
                    //flip all the corresponding mask upon the filled value
                    row_mask[i] |= (1 << (board[i][j] - '1'));
                    col_mask[j] |= (1 << (board[i][j] - '1'));
                    block_mask[getBlockIndex(i, j)] |= (1 << (board[i][j] - '1'));
                }
            }
        }
        //------------Search from the beginning of the board---------
        return dfs(board, 0, 0, row_mask, col_mask, block_mask, pos_to_char);
        //------------Run time 6ms----beats 76.24%------------------
    }

    //implement the search algorithm to find the right element to fill the board
    bool dfs(vector<vector<char>> &board, int i, int j, vector<int> &rows, 
            vector<int> &cols, vector<int> &blocks, unordered_map<int, char> &pos_to_char) {
        //checked all the rows already, return true
        if (i == 9) return true;
        //checked this row already, go to next row
        if (j == 9) return dfs(board, i + 1, 0, rows, cols, blocks, pos_to_char);
        //not an empty cell, go to next cell on the right
        if (board[i][j] != '.') 
            return dfs(board, i, j + 1, rows, cols, blocks, pos_to_char);
        int block_idx = getBlockIndex(i, j);
        //get all the available positions that we can fill the empty cell 
        //ex.   row[i] = 001000101   (at row i, there are already numbers 1, 3, and 7)
        //      col[j] = 100100000   (at col j, there are already numbers 6 and 9)
        //      blocks[block_idx] = 000010001  (at that blcok, there are number 1 and 5)
        //      when do | operation, we have 101110101, basically sum up all the occupied positions
        //      flip it we have 010001010, and also a 1 in the 32 bit
        //      truncate it by doing & with (1 << 9) - 1, we got the bit mask representing
        //      the available possitions that we can fill (this case it is 2, 4 and 8)
        int available_pos = ((1 << 9) - 1) & ~(rows[i] | cols[j] | blocks[block_idx]);
        while (available_pos != 0) {
            //low bit function, get the lowest 1 in the bit mask
            int low_bit = available_pos & -available_pos;
            //remove that from the bit mask
            available_pos ^= low_bit;
            //don't do a log2! that is implemented iteratively, using preprocess map
            board[i][j] = pos_to_char[low_bit];
            //update all the bit mask and of course the board
            rows[i] |= low_bit;
            cols[j] |= low_bit;
            blocks[block_idx] |= low_bit;
            if (dfs(board, i, j + 1, rows, cols, blocks, pos_to_char))
                return true;
            //reset bit mask and board
            rows[i] ^= low_bit;
            cols[j] ^= low_bit;
            blocks[block_idx] ^= low_bit;
            board[i][j] = '.';
        }
        //cannot find a number to fill in the possition, return false then
        return false;
    }

    //helper function to calculate the number of block one cell fit in
    //----- ----- -----
    //  0  |  1  |  2
    //----- ----- -----
    //  3  |  4  |  5
    //----- ----- -----
    //  6  |  7  |  8
    int getBlockIndex(int i, int j) {
        return i / 3 * 3 + j / 3;
    }


    /* 272. Closest Binary Search Tree Value II */
    /* Note: */
    /* Given target value is a floating point. */
    /* You may assume k is always valid, that is: k ≤ total nodes. */
    /* You are guaranteed to have only one unique set of k values in the BST that are closest to the target. */
    vector<int> closestKValues(TreeNode* root, double target, int k) {
        //as the hint suggested, we want to have two functions, get predecessor and get successor
        //they can be implemented in two ways: stack and recursion, here I choose to use stack to reduce the function call overhead
        //first find the lower, which is the TreeNode with value less than target
        stack<TreeNode *> less, greaterOrEqual;
        vector<int> res;
        auto p = root;
        while (p) {
            if (p -> val < target) {
                less.push(p);
                p = p -> right;
            }
            else {
                p = p -> left;
            }
        }
        p = root;
        while (p) {
            if (p -> val >= target) {
                greaterOrEqual.push(p);
                p = p -> left;
            }
            else {
                p = p -> right;
            }
        }
        while ((!less.empty() || !greaterOrEqual.empty()) && res.size() < k) {
            if (greaterOrEqual.empty() || (!less.empty() &&
                (target - less.top() -> val <= (double)greaterOrEqual.top() -> val - target))) {
                auto p = less.top();
                less.pop();
                res.push_back(p -> val);
                for (p = p -> left; p; p = p -> right) {
                    less.push(p);
                }
            }
            else {
                auto p = greaterOrEqual.top();
                greaterOrEqual.pop();
                res.push_back(p -> val);
                for (p = p -> right; p; p = p -> left) {
                    greaterOrEqual.push(p);
                }
            }
        }
        return res;
        //-----------------run time 16ms beats 65.78%----------------
    }

/*     425. Word Squares */
/*     Given a set of words (without duplicates), find all word squares you can build from them. */
/*     A sequence of words forms a valid word square if the kth row and column read the exact same string, */
/*     where 0 ≤ k < max(numRows, numColumns). */
    vector<vector<string>> wordSquares(vector<string>& words) {
        //suppose we already have n - 1 rows ready  and now looking for the n th row, what do we know about the n the row? 
        //the key observation is that the n the row must have a prefix that match the characters in the n th column start
        //from 0th row to n - 1 row
        //appended_string[0: n] = squares_sofar[0: n, n]
        //since it is a prefix search problem, we want to use trie tree to facillitate the searching process, first by injecting
        //all the strings into the root trienode and then use dfs and backtrack to find the right string with correct prefix
        auto root = new TrieNode_Map();
        vector<vector<string>> res;
        vector<string> list;
        if (words.size() == 0 || words[0].size() == 0)  return res;
        //get the dimension of the word
        const int dim = words[0].size();
        for (const string& word: words) {
            root = put(root, word, 0);
        }
        dfs(res, list, root, dim);
        //
        delete(root);
        return res;
        //-----------------run time 385ms----beats 20.30%--------
        //not optimized
    }

    //recursively put string into trienode
    TrieNode_Map* put(TrieNode_Map* root, const string& word, size_t pos) {
        if (!root)  root = new TrieNode_Map();
        if (pos == word.size()) {
            root -> val = word;
        }
        else {
            root -> children[word[pos]] = put(root -> children.find(word[pos]) == root -> children.end()? nullptr: root -> children[word[pos]], word, 
                    pos + 1);
        }
        return root;
    }

    void dfs(vector<vector<string>> &res, vector<string> &list, TrieNode_Map* root, const int dim) {
        size_t size = list.size();
        if (size == dim) {
            res.push_back(list);
            return;
        }
        auto p = root;
        //get prefix from list[0] to list.back
        for (auto it = list.begin(), last = list.end(); it != last; ++it) {
            if (p -> children.find((*it)[size]) == p ->children.end()) return;
            p = p -> children[(*it)[size]];
        }
        //now we know p is the starting root of the next word, search all it's children
        //here we use a bfs search to enable fast backtrack
        deque<TrieNode_Map*> q;
        q.push_back(p);
        while (!q.empty()) {
            auto front_node = q.front();
            q.pop_front();
            //we copy the map, in order to traverse it
            for (auto pair: front_node -> children) {
                if (pair.second -> val != "") {
                    list.push_back(pair.second -> val);
                    dfs(res, list, root, dim);
                    //backtrack
                    list.pop_back();
                }
                //otherwise push into the queue, continue bfs
                else {
                    q.push_back(pair.second);
                }
            }
        }
    }


    /* 440. K-th Smallest in Lexicographical Order */
    /* Given integers n and k, find the lexicographically k-th smallest integer in the range from 1 to n. */

    /* Note: 1 ≤ k ≤ n ≤ 10^9. */
    int findKthNumber(int n, int k) {
        //thoughts: given a prefix number: prefix, what is the number of numbers <= n that having this prefix?
        //start from prefix its self, it's going to be 1, and prefix * 10, there will be 10, prefix * 100, there will be 100
        //if 
        //after compute the total less number, added to the total count, if total count >= k, we know that the final result must be
        //with the prefix, make the prefix = prefix * 10 and add total count by 1. 
        //Repeat the process until we find a prefix with total count == k - 1
        int prefix = 1;
        int total_less = 0;
        while (total_less != k - 1) {
            int smallerWithPrefix = getSmallerWithPrefix(n, prefix);
            if (total_less + smallerWithPrefix >= k) {
                prefix *= 10;
                total_less += 1;
            }
            else {
                total_less += smallerWithPrefix;
                ++prefix;
            }
        }
        return prefix;
        //--------no running time info at the moment-------------
    }

    //helper function to calculate number of smaller or equal numbers with given prefix
    int getSmallerWithPrefix(long n, long prefix) {
        int multiplier = 1;
        int sum = 0;
        while (prefix * multiplier < n) {
            if (n - prefix * multiplier <= multiplier) {
                sum += (n - prefix * multiplier + 1);
                return sum;
            }
            else {
                sum += multiplier;
            }
            multiplier *= 10;
        }
        return sum;
    }

    /* 254. Factor Combinations */
    /* Numbers can be regarded as product of its factors. For example, */

    /*         8 = 2 x 2 x 2; */
    /*           = 2 x 4. */
    /* Write a function that takes an integer n and return all possible combinations of its factors. */
    vector<vector<int>> getFactors(int n) {
        //we start from 2, each time we only consider a factor that greater than the last factor
        //but less than equal to sqrt(rem)
        vector<vector<int>> res;
        vector<int> list;
        for (int i = 2; i * i <= n; ++i) {
            if (n % i == 0) {
                list.push_back(i);
                dfs(res, list, i, n / i);
                list.pop_back();
            }
        }
        return res;
        //---------run time 3ms, beats 49.86%----------
    }
    
    //dfs and backtrack
    void dfs(vector<vector<int>> &res, vector<int> &list, int last, int rem) {
        list.push_back(rem);
        res.push_back(list);
        list.pop_back();
        for (int next = last; next * next <= rem; ++next) {
            if (rem % next == 0) {
                list.push_back(next);
                dfs(res, list, next, rem / next);
                list.pop_back();
            }
        }
    }

    /* 239. Sliding Window Maximum */
    /* Given an array nums, there is a sliding window of size k which is moving from the very left of the array */
    /* to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. */

    /* For example, */
    /*     Given nums = [1,3,-1,-3,5,3,6,7], and k = 3. */
    vector<int> maxSlidingWindow(vector<int> &nums, int k) {
        //keeping a deque that keeping a monotomic decreasing sequence of the values in the window
        //everytime we move the deque, pop out the values in the front that is less than or equal to the new value
        //and remove the end value if it's index is out of the window scope, then the end value left should be the value
        //to be added
        vector<int> res;
        //a deque keep the index of the values in num within the window
        deque<int> window;
        if (k == 0 || nums.size() == 0) return res;
        //there is guaranteed to be nums.size() - k + 1 values
        res.reserve(nums.size() - k + 1);
        for (int i = 0; i < nums.size(); ++i) {
            //keep monotomic property
            while (!window.empty() && nums[window.front()] <= nums[i]) {
                window.pop_front();
            }
            //add the new value
            window.push_front(i);
            while (!window.empty() && window.back() <= i - k) {
                window.pop_back();
            }
            if (i >= k - 1) {
                res.push_back(nums[window.back()]);
            }
        }
        return res;
        //-----------run time 89ms, beats 38.23%-------------
    }

    /* 126. Word Ladder II */
    /* Given two words (beginWord and endWord), and a dictionary's word list, */ 
    /* find all shortest transformation sequence(s) from beginWord to endWord, such that: */

    /* Only one letter can be changed at a time */    
    /* Each intermediate word must exist in the word list */
    vector<vector<string>> findLadders(const string& beginWord, const string& endWord, unordered_set<string> &wordList) {
        //one of the hardest problem in LeetCode, build map by bfs, and then use dfs to find all the path
        unordered_map<string, unordered_set<string>> word_map;
        //create search queue
        unordered_set<string> search_queue{beginWord};
        wordList.erase(beginWord);
        wordList.insert(endWord);
        while (!search_queue.empty()) {
            //this set is going to be the next layer discovered by current layer
            unordered_set<string> neighbors;
            for (auto it = search_queue.begin(), end = search_queue.end();
                    it != end;
                    ++it) {
                string node(*it);
                //we go by each character and search it's 25 possible neighbors
                for (int pos = 0; pos < node.size(); ++pos) {
                    char temp = node[pos];
                    for (char c = 'a'; c <= 'z'; ++c) {
                        if (c == temp)  continue;
                        node[pos] = c;
                        if (wordList.find(node) != wordList.end()) {

                            //setup a path
                            word_map[*it].insert(node);
                            neighbors.insert(node);
                        }
                    }
                    //donot forget to reset the char value
                    node[pos] = temp;
                }
            }
            //already find the final word, stop bfs right away
            if (neighbors.find(endWord) != neighbors.end()) break;
            for (auto word: neighbors) {
                wordList.erase(word);
            }
            //wordList.erase(neighbors.begin(), neighbors.end());
            search_queue = neighbors;
        }
        //now we already have the map, do dfs to find all the possible path
        vector<vector<string>> res;
        vector<string> path {beginWord};
        dfs(beginWord, endWord, res, path, word_map);
        return res;
        //-----------run time 325ms betas 63.21%-------------------
    }

    //recursively search for the path
    void dfs(const string& current, const string& endWord, vector<vector<string>> &res, 
            vector<string> &path, unordered_map<string, unordered_set<string>> &word_map){
        if (current == endWord) {
            res.push_back(path);
            return;
        }
        for (auto neighbor: word_map[current]) {
            path.push_back(neighbor);
            dfs(neighbor, endWord, res, path, word_map);
            path.pop_back();
        }
    }

    //an alternative solution, using two end bfs
    //from https://discuss.leetcode.com/topic/16826/88ms-accepted-c-solution-with-two-end-bfs-68ms-for-word-ladder-and-88ms-for-word-ladder-ii/3

    vector<vector<string>> findLadders_perfect(string start, string end, unordered_set<string> &dict) {
        vector<vector<string> > ladders;
        vector<string> ladder;
        ladder.push_back(start);
        unordered_set<string> startWords, endWords;
        startWords.insert(start);
        endWords.insert(end);
        unordered_map<string, vector<string> > children;
        bool flip = true;
        if (searchLadders(startWords, endWords, dict, children, flip))
            genLadders(start, end, children, ladder, ladders);
        return ladders;
        /* /   -------run time 96ms betas 79.12%------------- */
    }

    bool searchLadders(unordered_set<string>& startWords, unordered_set<string>& endWords, 
            unordered_set<string>& dict, unordered_map<string, vector<string> >& children, bool flip) {
        flip = !flip;
        if (startWords.empty()) return false;
        if (startWords.size() > endWords.size())
            return searchLadders(endWords, startWords, dict, children, flip);
        for (string word : startWords) dict.erase(word);
        for (string word : endWords) dict.erase(word);
        unordered_set<string> intermediate;
        bool done = false;
        for (string word : startWords) {
            int n = word.length();
            string temp = word;
            for (int p = 0; p < n; p++) {
                char letter = word[p];
                for (int i = 0; i < 26; i++) {
                    word[p] = 'a' + i;
                    if (endWords.find(word) != endWords.end()) {
                        done = true;
                        flip ? children[word].push_back(temp) : children[temp].push_back(word);
                    }
                    else if (!done && dict.find(word) != dict.end()) {
                        intermediate.insert(word);
                        flip ? children[word].push_back(temp) : children[temp].push_back(word);
                    }
                }   
                word[p] = letter;
            }
        }
        return done || searchLadders(endWords, intermediate, dict, children, flip);
    }
    void genLadders(string& start, string& end, unordered_map<string, vector<string> >& children, 
            vector<string>& ladder, vector<vector<string> >& ladders) {
        if (start == end) {
            ladders.push_back(ladder);
            return;
        }
        for (string child : children[start]) {
            ladder.push_back(child);
            genLadders(child, end, children, ladder, ladders);
            ladder.pop_back();
        }
    }

    /* 444. Sequence Reconstruction */
    /* Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. */ 
    /* The org sequence is a permutation of the integers from 1 to n, with 1 ≤ n ≤ 104. */ 
    /* Reconstruction means building a shortest common supersequence of the sequences in seqs */ 
    /* (i.e., a shortest sequence so that all sequences in seqs are subsequences of it). */ 
    /* Determine whether there is only one sequence that can be reconstructed from seqs and it is the org sequence. */
    bool sequenceReconstruction(vector<int> &org, vector<vector<int>> &seqs) {
        //the first tuition of this problem is that it might require using of graph search but the right solution is not
        //since we need to determine is there a "unique" sequence. The adjacent elements in the original sequence must be
        //presented as adjacent subsequence and we need org.size() - 1 such "edges". So we can use edge counter to do the trick
        //the map to keep the element -> position pair, a simple way would be using vector though
        unordered_map<int, int> inversed_map;
        //see if a edge has been visited by checking its head element
        unordered_set<int> visited;
        if (seqs.size() == 0 || org.size() == 0)    return false;
        for (int idx = 0; idx < org.size(); ++idx) {
            inversed_map[org[idx]] = idx;
        }
        //edge counter
        int count_edge = 0;
        for (auto seq: seqs) {
            if (seq.size() != 0 && inversed_map.find(seq[0]) == inversed_map.end()) return false;
            for (int idx = 1; idx < seq.size(); ++idx) {
                //if any element is not showing up, or it forms a self-loop, or it is a inversed pair, return false
                if (seq[idx - 1] == seq[idx] || inversed_map.find(seq[idx]) == inversed_map.end() ||
                        inversed_map[seq[idx - 1]] > inversed_map[seq[idx]])  return false;
                //check condition to increment counter
                if (inversed_map[seq[idx - 1]] + 1 == inversed_map[seq[idx]] 
                        && visited.find(inversed_map[seq[idx - 1]]) == visited.end()) {
                    ++count_edge;
                    visited.insert(inversed_map[seq[idx - 1]]);
                }
            }
        }
        return count_edge == org.size() - 1;
    }

    /* 446. Arithmetic Slices II - Subsequence */
    /* A sequence of numbers is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same. */
    int numberOfArithmeticSlices(vector<int>& A) {
        //essentially dp + memorization
        //keep a map for each position, maintain diff -> count of sequence mapping
        //each time we see a new number, compare that with all the previous numbers, increment count if that previous number contains the same difference key
        int res = 0;
        if (A.size() == 0)  return 0;
        vector<unordered_map<long long, int>> dp(A.size());
        long long diff = 0;
        for (int i = 1; i < A.size(); ++i) {
            for (int j = i - 1; j >= 0; --j) {
                diff = static_cast<long long>(A[i]) - static_cast<long long>(A[j]);
                if (dp[j].count(diff)) {
                    res += dp[j][diff];
                    dp[i][diff] += dp[j][diff];
                }
                ++ dp[i][diff];
            }
        }
        return res;
    }

    /* 466. Count The Repetitions */
    /* Define S = [s,n] as the string S which consists of n connected strings s. For example, ["abc", 3] ="abcabcabc". */

    /* On the other hand, we define that string s1 can be obtained from string s2 if we can remove some characters from s2 */
    /* such that it becomes s1. For example, “abc” can be obtained from “abdbec” based on our definition, but it can not be obtained from “acbbe”. */

    /* You are given two non-empty strings s1 and s2 (each at most 100 characters long) and two integers 0 ≤ n1 ≤ 106 and 1 ≤ n2 ≤ 106. */
    /* Now consider the strings S1 and S2, where S1=[s1,n1] and S2=[s2,n2]. Find the maximum integer M such that [S2,M] can be obtained from S1. */
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
        //------------run time 3ms---------beats 87%---------------------//
    }
    /* 465. Optimal Account Balancing */
    /* A group of friends went on holiday and sometimes lent each other money. */ 
    /* For example, Alice paid for Bill's lunch for 0. Then later Chris gave Alice $5 for a taxi ride. */ 
    /* We can model each transaction as a tuple (x, y, z) which means person x gave person y $z. */ 
    /* Assuming Alice, Bill, and Chris are person 0, 1, and 2 respectively (0, 1, 2 are the person's ID), */ 
    /* the transactions can be represented as [[0, 1, 10], [2, 0, 5]]. */

    /* Given a list of transactions between a group of people, return the minimum number of transactions required to settle the debt. */
    int minTransfers(vector<vector<int>>& transactions) {
        unordered_map<int, int> balances;
        for (auto& transaction: transactions) {
            int first = transaction[0], second = transaction[1], debt = transaction[2];
            if (balances.find(first) == balances.end()) balances[first] = 0;
            if (balances.find(second) == balances.end()) balances[second] = 0;
            balances[first] += debt;
            balances[second] -= debt;
        }
        vector<int> creditors;
        vector<int> debtors;
        for (auto it = balances.begin(); it != balances.end(); ++it) {
            if (it->second > 0) {
                creditors.push_back(it->second);
            }
            else if(it->second < 0) {
                debtors.push_back(-it->second);
            }
        }
        sort(creditors.begin(), creditors.end());
        sort(debtors.begin(), debtors.end());

        int matchedCount = 0;
        for (int i = 0; i < creditors.size(); ++i) {
            int count = 0;
            if (getMatchedCount(creditors, debtors, i, 0, count))
                matchedCount += count;
        }
        for (int i = 0; i < debtors.size(); ++i) {
            int count = 0;
            if (getMatchedCount(debtors, creditors, i, 0, count)) 
                matchedCount += count;
        }
        //match the rest transfers
        int i = 0, j = 0;
        while (i < creditors.size() && j < debtors.size()) {
            if (creditors[i] == 0) {
                ++i;
                continue;
            }
            if (debtors[j] == 0) {
                ++j;
                continue;
            }
            if (creditors[i] < debtors[j]) 
                debtors[j] -= creditors[i++];
            else if (creditors[i] > debtors[j])
                creditors[i] -= debtors[j++];
            else {
                creditors[i++] = 0;
                debtors[j++] = 0;
            }
            ++matchedCount;
        }
        return matchedCount;
        //-----------run time 0ms, beat 85.13%---------------------
    }

    bool getMatchedCount(vector<int>& creditors, vector<int>& debtors, int pos_c, int pos_d, int& count) {
        if (creditors[pos_c] == 0)  return true;
        if (pos_d >= debtors.size() || debtors[pos_d] > creditors[pos_c]) return false;
        //either take d or not take d
        //take pos_d
        if (debtors[pos_d] != 0) {
            creditors[pos_c] -= debtors[pos_d];
            int temp_debt = debtors[pos_d];
            debtors[pos_d] = 0;
            if (getMatchedCount(creditors, debtors, pos_c, pos_d + 1, ++count)) return true;
            else {
                debtors[pos_d] = temp_debt;
                creditors[pos_c] += debtors[pos_d];
                --count;
            }
        }
        return getMatchedCount(creditors, debtors, pos_c, pos_d + 1, count);
    }
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

/* 381. Insert Delete GetRandom O(1) - Duplicates allowed */
/* Design a data structure that supports all following operations in average O(1) time. */

/* Note: Duplicate elements are allowed. */
/* insert(val): Inserts an item val to the collection. */
/*              remove(val): Removes an item val from the collection if present. */
/*              getRandom: Returns a random element from current collection of elements. */ 
/*                         The probability of each element being returned is linearly related */
/*                         to the number of same value the collection contains. */
class RandomizedCollection {
    //Keep a vector to hold all the elements that we current have (this will allow O(1) getRandom operation)
    //Keep a hashmap to hold use element value as key, a set containing all the indexes in the vector as value
    //when we remove a value, we get its index, swap with the last index in the array(update hashmap of course)
    //and then pop_back the vector
    //
    private:
        unordered_map<int, unordered_set<int>> val_to_index;
        vector<int> vals;

    public:
        RandomizedCollection() {
        }

        bool insert (int val) {
            vals.push_back(val);
            val_to_index[val].insert(vals.size() - 1);
            return val_to_index[val].size() == 1;
        }

        bool remove (int val) {
            if (val_to_index.find(val) == val_to_index.end())   return false;
            //if the val turns out to be the very last, it becomes easy
            if (vals.back() == val) {
                val_to_index[val].erase(vals.size() - 1);
                if (!val_to_index[val].size())  val_to_index.erase(val);
                vals.pop_back();
            }
            else {
                //now we need to get two set and swap the index
                auto &last_element_set = val_to_index[vals.back()];
                auto &remove_element_set = val_to_index[val];
                int remove_index = *remove_element_set.begin();
                int last_index = vals.size() - 1;
                last_element_set.insert(remove_index);
                last_element_set.erase(last_index);
                remove_element_set.erase(remove_index);
                if (!remove_element_set.size()) val_to_index.erase(val);
                vals[remove_index] = vals.back();
                vals.pop_back();
            }
            return true;
        }

        int getRandom() {
            int idx = rand() % vals.size();
            return vals[idx];
        }
        //------run time 129ms ---- beats 26.93%---------//
};

/* 211. Add and Search Word - Data structure design */
/* Design a data structure that supports the following two operations: */
class WordDictionary {
    //using a trie is obvious, but should we use a trie implemented using vector of children
    //or unordered_map of children
    //Here I choose to use map version since it will be less useless operation when we
    //search by '.'
    private:
        TrieNode_Map* root;

        //recursively add
        TrieNode_Map *put(TrieNode_Map *node, const string& word, int pos) {
            if (!node)  node = new TrieNode_Map();
            if (pos != word.size()) {
                node -> children[word[pos]] = put(node -> children.find(word[pos]) == 
                    node -> children.end()? nullptr: node -> children[word[pos]],
                    word, pos + 1);
            }
            else {
                //add terminater
                node -> val = "#";
            }
            return node;
        }       

        //recursively search
        bool search(TrieNode_Map *node, const string& word, int pos) {
            if (pos == word.size()) return node -> val == "#";
            if (word[pos] != '.') {
                return node -> children.find(word[pos]) == node -> children.end()?
                    false: search(node ->children[word[pos]], word, pos + 1);
            }
            else {
                //go through all the children and check if there is a match
                for (auto pair: node -> children) {
                    if (search(pair.second, word, pos + 1)) return true;
                }
            }
            return false;
        }

    public:
        //ctor
        WordDictionary():root(new TrieNode_Map){}

        //this dictionary has reference semantic, do not allow copy
        WordDictionary(const WordDictionary& other) = delete;
        WordDictionary operator=(const WordDictionary& other) = delete;

        //dtor
        ~WordDictionary(){
            delete(root);
        }

        //Adds a word into the data structure
        void addWord(string word) {
            put (root, word, 0);
        }

        // Returns if the word is in the data structure. A word could
        // contain the dot character '.' to represent any one letter.
        bool search(const string& word) {
            return search(root, word, 0);
        }
};

/* 460. LFU Cache */
/* Design and implement a data structure for Least Frequently Used (LFU) cache. */ 
/* It should support the following operations: get and put. */

/* get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1. */
/* put(key, value) - Set or insert the value if the key is not already present. */ 
/* When the cache reaches its capacity, it should invalidate the least frequently used item before inserting a new item. */ 
/* For the purpose of this problem, when there is a tie (i.e., two or more keys that have the same frequency), */
/* the least recently used key would be evicted */
class LFUCache {
    public:
        explicit LFUCache (int capacity): cap(capacity), pivot(new LFUNode(0, 0, -1)){
            pivot->prev = pivot;
            pivot->next = pivot;
        }
        
        int get(int key) {
            if (key_to_node.find(key) == key_to_node.end())
                return -1;
            LFUNode* node = key_to_node[key];
            updateNode(node);
            return node->value;
        }

        void put(int key, int value) {
            if (cap == 0)   return;
            //no key presented, insert at the end
            if (key_to_node.find(key) == key_to_node.end()) {
                LFUNode* tail = new LFUNode(key, value, 0);
                key_to_node[key] = tail;
                if (key_to_node.size() > cap) {
                    LFUNode* old_tail = pivot->prev;
                    key_to_node.erase(old_tail->key);
                    unlink(old_tail);
                    if (freq_to_node[old_tail->freq] == old_tail) {
                        freq_to_node.erase(old_tail->freq);
                    }
                    delete old_tail;
                }
                link(tail, pivot);
                updateNode(tail);
            }
            //otherwise update value
            else {
                key_to_node[key]->value = value;
                updateNode(key_to_node[key]);
            }
        }

        LFUCache(const LFUCache& other) = delete;
        LFUCache& operator=(const LFUCache& rhs) = delete;
        ~LFUCache() {
            //delete all node in doubly linked list
            auto node_ptr = pivot->next;
            while (node_ptr != pivot) {
                auto temp = node_ptr->next;
                delete node_ptr;
                node_ptr = temp;
            }
            delete pivot;
        }

    private:
        struct LFUNode {
            LFUNode* prev;
            LFUNode* next;
            int key;
            int value;
            int freq;
            LFUNode(int k, int v, int f): prev(nullptr), next(nullptr), key(k), value(v), freq(f) {}
        };

        //frequency to node, if two node ties in frequency, save the preivous one in doubly linkedlist
        unordered_map<int, LFUNode*> freq_to_node;
        //key to node mapping
        unordered_map<int, LFUNode*> key_to_node;
        //doubly linked list(circular), initialize with one node
        LFUNode* pivot;

        const int cap;

        void unlink(LFUNode* node) {
            if (node->prev) {
                node->prev->next = node->next;
                node->next->prev = node->prev;
            }
        }

        void link(LFUNode* node, LFUNode* behind) {
            node->next = behind;
            node->prev = behind->prev;
            behind->prev->next = node;
            behind->prev = node;
        }

        void updateNode(LFUNode* node) {
            if (freq_to_node.find(node->freq) != freq_to_node.end()) {
                //two senario, its the node itself, or someother node
                if (freq_to_node[node->freq] == node) {
                   if (node->next->freq == node->freq) {
                       freq_to_node[node->freq] = node->next;
                   }
                   else {
                       freq_to_node.erase(node->freq);
                   }
                }
                else {
                    unlink(node);
                    link(node, freq_to_node[node->freq]);
                }
            }
            ++node->freq;
            if (freq_to_node.find(node->freq) != freq_to_node.end()) {
                unlink(node);
                auto temp = freq_to_node[node->freq];
                link(node, temp);
            }
            freq_to_node[node->freq] = node;
        }
        //------------run time 109ms, beats 96.00%------------------
};

int main() {
    /* LFUCache cache(2); */
    /* cache.put(1, 1); */
    /* cache.put(2, 2); */
    /* cout << cache.get(1) << endl;       // returns 1 */
    /* cache.put(3, 3);    // evicts key 2 */
    /* cout << cache.get(2) << endl;       // returns -1 (not found) */
    /* cout << cache.get(3) << endl;       // returns 3. */
    /* cache.put(4, 4);    // evicts key 1. */
    /* cout << cache.get(1) << endl;       // returns -1 (not found) */
    /* cout << cache.get(3) << endl;       // returns 3 */
    /* cout << cache.get(4) << endl;       // returns 4 */
    LFUCache cache(3);
    cache.put(2, 2);
    cache.put(1, 1);
    cout << cache.get(2) << endl;       // returns 1
    cout << cache.get(1) << endl;       // returns 1
    cout << cache.get(2) << endl;       // returns 1
    cache.put(3, 3);    // evicts key 2
    cache.put(4, 4);    // evicts key 1.
    cout << cache.get(3) << endl;       // returns 3.
    cout << cache.get(2) << endl;       // returns -1 (not found)
    cout << cache.get(1) << endl;       // returns -1 (not found)
    cout << cache.get(4) << endl;       // returns 4
}
