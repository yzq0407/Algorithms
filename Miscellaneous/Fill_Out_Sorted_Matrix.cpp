//Google onsite interview, second round 10/12/2016

#include <iostream>
#include <vector>
#include <unordered_map>


//given a 5 x 5 matrix, fill in numbers from 1 - 25 (inclusive, no duplicate), such that:
// all the rows and columns are sorted
//output number of distinct solution
//

using namespace std;

class Solution {
private:
    typedef long long ll;
    //dimension, in this question it is 5 (we assume it is less than 16, so that we can use at most 4 bit to represent it)
    //actually when number is greater than 5, even long long will overflow!, the number of solution grows super exponentially 
    int dimension;

    //dp matrix, keep number of solutions associated with filling (n1, n2, n3, n4, n5) number of entries
    //in each row
    //ex. (4, 4, 2, 1, 0) means number of solutions when filling 4 entries in the first row, 4 entries in the second row
    //2 entries in the third row, 1 entries in the fourth row and no entires in the last row with number 1 - 11 inclusive
    //the key in the hashmap is a hashed int associated with the tuple (this will enable faster lookup and hashing)
    unordered_map<ll, int> dp;

    //hash the tuple such that we can get a unique long long value to represent the tuple
    ll hash_tuple (const vector<int> &tuple) {
        ll hash_key = 0;
        //using bit wise shift to hash it, based on the assumption that num in tuple is less than 16, we can combine them
        //into a long long value
        for (int num: tuple) {
            hash_key = (hash_key << 4) ^ num; 
        }
        return hash_key;
    }

    //search down the matrix to generate a valid tuple
    void search_tuple(int filled, int row, vector<int>& tuple) {
        //fill this row with "filled" number of entries and search the next row
        tuple[row] = filled;
        //if row is already the last one, find all its neighbors and update dp
        if (row == dimension - 1) {
            int total = 0;
            for (int off_row = 0; off_row < dimension; ++off_row) {
                //we want to subtract one from this row in the tuple such that it is a neighbor, but we also want to 
                //maintain the tuple's property
                if (tuple[off_row] > 0 && (off_row == dimension - 1 || tuple[off_row] > tuple[off_row + 1])) {
                    --tuple[off_row];
                    total += dp[hash_tuple(tuple)];
                    ++tuple[off_row];
                }
            }
            dp[hash_tuple(tuple)] = total;
            return;
        }
        //else, recursively fill next row
        //remember, next row is always less than or equal to this row
        for (int next = 0; next <= filled; ++next) {
            search_tuple(next, row + 1, tuple);
        }
        
    }

public:
    Solution(int d = 5): dimension(d) {
        //it is important that there is 1 solution if we choose not to fill anything!
        dp[0] = 1;
    }

    ll solve() {
        //a tuple to represent 
        vector<int> filled_entry(dimension, 0);
        //for any given tuple (n1, n2, n3, n4, n5), number of solutions will be:
        //S(n1, n2, n3, n4, n5) = S(n1 - 1, n2, n3, n4, n5) + S(n1, n2 - 1, n3, n4, n5) + S(n1, n2, n3 -1, n4, n5)
        // + S(n1, n2, n3, n4 - 1, n5) + S(n1, n2, n3, n4, n5 - 1)
        //Also it need to subject to the rule that any valid tuple must satisfy n1 >= n2 >= n3 >= n4 >= n5
        vector<int> tuple(dimension);
        for (int filled = 1; filled <= dimension; ++filled) {
            search_tuple(filled, 0, tuple);
        }
        return dp[hash_tuple(vector<int>(dimension, dimension))];
    }
    
};


int main () {
    //even overflow happens when dim >= 6, we can still get an idea about how fast this solution is
    //a dfs solution is trivial, which takes more than 15mins in my laptop to compute all the solutions in a 5x5 matrix!
    for (int dim = 1; dim <= 8; ++dim) {
        Solution s(dim);
        cout << "For " << dim<< " x " << dim <<" matrix, number of distinct solutions is: " << s.solve() << endl;
    }
}

