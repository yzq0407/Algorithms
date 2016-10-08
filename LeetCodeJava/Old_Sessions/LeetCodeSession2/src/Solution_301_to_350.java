import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;

public class Solution_301_to_350 {
	
	
	public int maxCoins(int[] nums) {
		if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1)
            return nums[0];
		//dp[i][j] is the max coins get for the range i -- j
		//which can be calculated by letting k = i -- j as the last balloon to be break
		//then we know that
		//dp[i][j] = max(dp[i][k] + dp[k][j] + nums[k]) for every k
		int n = nums.length;

		int[][] dp = new int[n][n];
		dp[0][0] = nums[0] * nums[1];
		dp[n - 1][n - 1] = nums[n - 2] * nums[n - 1];
		for (int i = 1; i < n - 1; i++) {
			dp[i][i] = nums[i - 1] * nums[i] * nums[i + 1];
		}
		for (int span = 1; span < n; span++) {
			for (int start = 0; start + span < n; start++) {
				int end = start + span;
				for (int k = start; k <= end; k++) {
					//  start --- k - 1
					// k + 1----end
					int left = (start <= k-1)? dp[start][k - 1]: 0;
					int right = (k + 1 <= end)? dp[k + 1][end]: 0;
					int lBounder = (start == 0)? 1: nums[start - 1];
					int rBounder = (end == n - 1)? 1: nums[end + 1];
					dp[start][end] = Math.max(left + right + lBounder * rBounder *nums[k],
							dp[start][end]);
				}
			}
		}
		return dp[0][n - 1];
    }
	
	public List<Integer> topKFrequent(int[] nums, int k) {
		if (nums.length == 0) {
			return new ArrayList<Integer>();
		}
		HashMap<Integer, Integer> count = new HashMap<>();
        for (int num: nums) {
            int freq = count.containsKey(num)? count.get(num) + 1: 1;
            count.put(num, freq);
        }
        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>(){

			@Override
			public int compare(int[] o1, int[] o2) {
				// TODO Auto-generated method stub
				return o1[1] - o2[1];
			}
        	
        });
        for (int num: count.keySet()) {
        	if (pq.size() < k) {
        		pq.offer(new int[]{num, count.get(num)});
        	}
        	else {
        		int freq = pq.peek()[2];
        		if (count.get(num) > freq) {
        			pq.poll();
        			pq.offer(new int[]{num, count.get(num)});
        		}
        	}
        }
        LinkedList<Integer> list = new LinkedList<>();
        while (!pq.isEmpty()) {
        	list.addFirst(pq.poll()[0]);
        }
        return list;
    }
	
	public class KVPair {
		int key;
		int value;
		KVPair(int key, int value) {
			this.key = key;
			this.value = value;
		}
	}

	
	public int compareNumber(LinkedList<Integer> num1, LinkedList<Integer> num2) {
		assert(num1.size() == num2.size());
		Iterator<Integer> it1 = num1.iterator();
		Iterator<Integer> it2 = num2.iterator();
		while (it1.hasNext() && it2.hasNext()) {
			int n1 = it1.next();
			int n2 = it2.next();
			if (n1 != n2) {
				return n1 - n2;
			}
		}
		return 0;
	}
	
	public void findMaxNumber(int[] num, int k) {
		if (k == 0) {
			
		}
	}
	
	public int[] countBits(int num) {
        int[] result = new int[num + 1];
        result[0] = 0;
        populate(1, result, 1);
        return result;
    }
	
	
	public void populate(int n, int[] array, int bits) {
		if (n >= array.length) {
			return;
		}
		array[n] = bits;
		populate(n<<1, array, bits);
		populate((n<<1) + 1, array, bits + 1);
	}
	
	public List<Integer> countSmaller(int[] nums) {
		List<Integer> ret = new ArrayList<>();
		if (nums == null || nums.length == 0) {
			return ret;
		}
		int[] sorted = Arrays.copyOf(nums, nums.length);
		Arrays.sort(sorted);
		HashMap<Integer, Integer> args = new HashMap<>();
		for (int i = 0; i < nums.length; i++) {
			if (!args.containsKey(sorted[i])) {
				args.put(sorted[i], i);
			}
		}
		FenwickTree ft = new FenwickTree(nums.length);
		int[] aux = new int[nums.length];
		for (int i = sorted.length - 1; i >= 0; i--) {
			int pivot = args.get(nums[i]);
			aux[i] = ft.sum(pivot);
			ft.add(pivot, 1);
		}
		for (int num: aux) {
			ret.add(num);
		}
		return ret;
		// [5, 1, -2, 0]
		//after argsort
		// [2, 3, 1, 0]
		//so what we want to do is to add from the last element
		// 0 is the second smallest, after adding, it is
		//[0, 1, 0, 0] -----> presum = 0
		// -2 is smallest
		//[1, 1, 0, 0] -----> presum = 0
		// 1 is the third largest
		//[1, 1, 1, 0] ----->presum = 2
		//then the largest 5
		//[1, 1, 1, 1] -----> presum = 3
    }
	
	public static class ArrayUtil {
	    public static int[] argsort(final int[] a) {
	        return argsort(a, true);
	    }

	    public static int[] argsort(final int[] a, final boolean ascending) {
	        Integer[] indexes = new Integer[a.length];
	        for (int i = 0; i < indexes.length; i++) {
	            indexes[i] = i;
	        }
	        Arrays.sort(indexes, new Comparator<Integer>() {
	            @Override
	            public int compare(final Integer i1, final Integer i2) {
	                return (ascending ? 1 : -1) * Integer.compare(a[i1], a[i2]);
	            }
	        });
	        return asArray(indexes);
	    }

	    public static <T extends Number> int[] asArray(final T... a) {
	        int[] b = new int[a.length];
	        for (int i = 0; i < b.length; i++) {
	            b[i] = a[i].intValue();
	        }
	        return b;
	    }

	}
	
	public static class FenwickTree {
		int[] tree;
		int[] array;
		
		FenwickTree(int n) {
			tree = new int[n + 1];
			array = new int[n];
		}
		
		FenwickTree(int[] array) {
			tree = new int[array.length + 1];
			this.array = new int[array.length];
			for (int i = 0; i < array.length; i++) {
				change(i, array[i]);
			}
		}
		
		public int sum(int n) {
			int sum = 0;
			while (n > 0) {
				sum += tree[n];
				n -= (n & -n);
			}
			return sum;
		}
		
		public void add(int i, int addition) {
			array[i] += addition;
			i = i + 1;
			while (i <= array.length) {
				tree[i] += addition;
				i += (i & -i);
			}
		}
		
		public void change(int i, int value) {
			int diff = value - array[i];
			array[i] = value;
			i = i + 1;
			while (i <= array.length) {
				tree[i] += diff;
				i += (i & -i);
			}
		}
	}
	
	public int minPatches(int[] nums, int n) {
		int range = 0;
		int count = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] <= range + 1) {
				range += nums[i];
				if (range >= n) {
					return count;
				}
			}
			else {
				count++;
				range += (range + 1);
				--i;
			}
		}
		while (range < n) {
			count++;
			if (range < range * 2 + 1)
				range += (range + 1);
			else
				return count;
		}
		return count;
    }
	
//	public static class NumMatrix {
//	    int[][] sum;
//	    public NumMatrix(int[][] matrix) {
//	    	if (matrix.length == 0) {
//	    		sum = new int[1][1];
//	    	}
//	        int n = matrix.length;
//	        int m = matrix[0].length;
//	        sum = new int[n + 1][m + 1];
//	        for (int i = 0; i < n; i++) {
//	            for (int j = 0; j < m; j++) {
//	                sum[i + 1][j + 1] = sum[i][j + 1] + sum[i + 1][j] - sum[i][j] + matrix[i][j]; 
//	            }
//	        }
//	        for (int i = 0; i <= n; ++i) {
//	        	System.out.println(Arrays.toString(sum[i]));
//	        }
//	    }
//
//	    public int sumRegion(int row1, int col1, int row2, int col2) {
//	        return sum[row2 + 1][col2 + 1] - sum[row2 + 1][col1] - sum[row1][col2 + 1] + sum[row1][col1];
//	    }
//	}
	
	public class NestedIterator implements Iterator<Integer> {
	    Stack<Iterator<NestedInteger>> s;
	    Iterator<NestedInteger> current;
	    boolean hasNext;
	    Integer last = null;
	    
	    public NestedIterator(List<NestedInteger> nestedList) {
	    	s = new Stack<>();
	        current = nestedList.iterator();
	        move();
	    }

	    @Override
	    public Integer next() {
	        int temp = last;
	        last = null;
	        move();
	        return temp;
	    }
	    
	    public void move() {
	    	while (!s.isEmpty() || current.hasNext()) {
	        	if (current.hasNext()) {
	        		NestedInteger ni = current.next();
	        		if (ni.isInteger()) {
	        			last = ni.getInteger();
	        		}
	        		else {
	        			s.push(current);
	        			current = ni.getList().iterator();
	        		}
	        	}
	        	else {
	        		current = s.pop();
	        	}
	        }
	    }
	    
	    @Override
	    public boolean hasNext() {
	    	return last == null;
	    }
	}
	
	public int nthSuperUglyNumber(int n, int[] primes) {
        int[] pointers = new int[primes.length];
        int[] uglies = new int[n];
        uglies[0] = 1;
        PriorityQueue<Meta> pq = new PriorityQueue<>(new Comparator<Meta> () {
            public int compare(Meta m1, Meta m2) {
                return uglies[m1.p] * m1.prime - uglies[m2.p] * m2.prime;
            }
        });
        for (int prime: primes) {
            pq.offer(new Meta(0, prime));
        }
        for (int i = 1; i < n; i++) {
            Meta m = pq.peek();
            uglies[i] = uglies[m.p] * m.prime;
            while (uglies[m.p] * m.prime == uglies[i]) {
                pq.poll();
                m.p++;
                pq.offer(m);
                m = pq.peek();
            }
        }
        return uglies[n - 1];
    }
    
    public class Meta {
        int p;
        int prime;
        
        public Meta(int p, int prime) {
            this.p = p;
            this.prime = prime;
        }
    }
    
    public int maxProfit(int[] prices) {
    	if (prices.length == 0) {
    		return 0;
    	}
        int[] buy = new int[prices.length];
        int[] sell = new int[prices.length];
        buy[0] = -prices[0];
        for (int i = 1; i < prices.length; ++i) {
            //buy must be done after 1 day of sell, buy is 
            //buy[i] = Math.max(sell[i - 2] - prices[i], buy[i - 1])
            int prev = i >= 2? sell[i - 2]: 0;
            buy[i] = Math.max(prev - prices[i], buy[i - 1]);
            sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
        }
        return sell[prices.length - 1];
    }
    
    public static class NumMatrix {
        int[][] trees;
        int[][] m;

        public NumMatrix(int[][] matrix) {
            if (matrix.length == 0) {
                return;
            }
            trees = new int[matrix.length + 1][matrix[0].length + 1];
            m = new int[matrix.length][matrix[0].length];
            for (int i = 0; i < matrix.length; ++i) {
                for (int j = 0; j < matrix[0].length; ++j) {
                    update(i, j, matrix[i][j]);
                }
            }
            
        }

        public void update(int row, int col, int val) {
            int diff = val - m[row][col];
            m[row][col] += diff;
            for (int i = row + 1; i < trees.length; i += (i & -i)) {
                for (int j = col + 1; j < trees[0].length; j += (j & -j)) {
                    trees[i][j] += diff;
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return sum(row2, col2) - sum(row2, col1 - 1) - sum(row1 - 1, col2) + sum(row1 - 1, col1 - 1);
        }
        
        public int sum(int row, int col) {
            int sum = 0;
            for (int i = row + 1; i > 0; i -= (i & - i)) {
                for (int j = col + 1; j > 0; j -= (j & -j)) {
                    sum += trees[i][j];
                }
            }
            return sum;
        }
    }
    
    public List<String> removeInvalidParentheses(String s) {
        List<String> rv = new ArrayList<>();
        bfs(new HashSet<String>(), rv);
        return rv;
    }
    
    public void bfs(Set<String> set, List<String> list) {
        if (!list.isEmpty()) {
            return;
        }
        Set<String> newSet = new HashSet<>();
        for (String str: set) {
            if (isValid(str)){
                list.add(str);
            }
            else {
                for (int i = 0; i < str.length(); ++i) {
                    newSet.add(str.substring(0, i) + str.substring(i + 1));
                }
            }
        }
        bfs(newSet, list);
    } 
    public boolean isValid (String s) {
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(')
                count++;
            if (s.charAt(i) == ')')
                count--;
            if (count < 0) {
                return false;
            }
        }
        return count == 0;
    }
    
    HashMap<String, PriorityQueue<String>> map =new HashMap<>();
    List<String> route= new ArrayList<>();
    
    public List<String> findItinerary(String[][] tickets) {
    	for (String[] ticket: tickets) {
    		map.computeIfAbsent(ticket[0], k -> new PriorityQueue<String> ()).add(ticket[1]);
    	}
    	visit("JFK");
    	return route;
    }
   
    public void visit(String port) {
    	while (map.containsKey(port) && !map.get(port).isEmpty()) {
    		visit(map.get(port).poll());
    	}
    	route.add(0,  port);
    }
    
//    public int countRangeSum(int[] nums, int lower, int upper) {
//        int[] rangeSum = new int[nums.length + 1];
//        for (int i = 1; i <= nums.length; ++i) {
//        	rangeSum[i] = rangeSum[i - 1] + nums[i - 1];
//        }
//        //let's call it r[i] = nums[0] + ... r[i -1] so sum[ i -- j] =r[j + 1] - r[i]
//        //lower <= r[j + 1] - r[i] <= upper   lower - r[j + 1] <= -r[i] <= upper - r[j + 1]
//        // r[j + 1] - lower >= r[i] >= r[j + 1] - upper
//        // for a given j, we find i that satisfy i <= j and that
//        // start form j = 0 from i in two bounds and sum up that in the fenwick tree
//        int[] sortedNums = Arrays.copyOf(rangeSum, rangeSum.length);
//        Arrays.sort(sortedNums);
//        HashMap<Integer, Integer> map = new HashMap<> ();
//        for (rangeSum)
//        
//    }
    
    public String removeDuplicateLetters(String s) {
        int[] lastAppearance = new int[128];
        boolean[] contains = new boolean[128];
        for (int i = 0; i < s.length(); i++) {
            lastAppearance[s.charAt(i)] = i;
        }
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!contains[c]) {
                while (!stack.isEmpty() && lastAppearance[stack.peek()] > i && stack.peek() > c) {
                    contains[stack.pop()] = false;
                }
                stack.push(c);
                contains[c] = true;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (Character c: stack) {
            sb.append(c);
        }
        return sb.toString();
    }
    
    
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int coin: coins)   {
            if (coin <= amount) dp[coin] = 1;
        }
        for (int i = 1; i <= amount; ++i) {
        	//if (dp[i] != Integer.MAX_VALUE)	continue;
            for (int coin: coins) {
                if (i > coin && dp[i - coin] != Integer.MAX_VALUE) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE? -1: dp[amount];
    }
    
    
    public List<String> generateAbbreviations(String word) {
        List<String> list = new ArrayList<>();
        for (int i = 0; i < 1<<word.length(); ++i) {
        	//System.out.println(i);
            StringBuilder sb = new StringBuilder();
            int j = word.length() - 1;
            int countTrailing = 0;
            while (j >= 0) {
                int mask = 1 << j;
                if ((i & mask) == 0) {
                    if (countTrailing != 0) sb.append(String.valueOf(countTrailing));
                    countTrailing = 0;
                    sb.append(word.charAt(word.length() - j - 1));
                }
                else {
                    countTrailing++;
                }
                j--;
            }
            if (countTrailing != 0) sb.append(String.valueOf(countTrailing));
            //System.out.println(sb.toString());
            list.add(sb.toString());
        }
        return list;
    }
    
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for (int num: nums1) sb1.append(String.valueOf(num));
        for (int num: nums2) sb2.append(String.valueOf(num));
        String s1 = sb1.toString(), s2 = sb2.toString();
        // dp[i, j] == maximum string with length j by using first i characters
        //dp[i, j] = max(dp[i - 1, j], dp[i - 1, j - 1] + s[i]);
        String[] comb1 = fillDP(s1, k);
        System.out.println(Arrays.toString(comb1));
        String[] comb2 = fillDP(s2, k);
        System.out.println(Arrays.toString(comb2));
        String max = "";
        for (int i = 0; i <= k; i++) {
        	int j = k - i;
        	if (i >= comb1.length || j >= comb2.length)	continue;
        	//if (j >= comb2.length)	continue;
        	String sub1 = comb1[i];
        	String sub2 = comb2[j];
        	StringBuilder current = new StringBuilder();
        	int p1 = 0;
        	int p2 = 0;
        	while (p1 < sub1.length() && p2 < sub2.length()) {
        		char c1 = sub1.charAt(p1);
        		char c2 = sub2.charAt(p2);
        		if (sub1.substring(p1).compareTo(sub2.substring(p2)) > 0) {
        			current.append(c1);
        			p1++;
        		}
        		else {
        			current.append(c2);
        			p2++;
        		}
        	}
        	while (p1 < sub1.length())	current.append(sub1.charAt(p1++));
        	while (p2 < sub2.length())	current.append(sub2.charAt(p2++));
        	String result = current.toString();
        	if (result.compareTo(max) > 0)	max = result;
        }
        int[] rv = new int[max.length()];
        for (int i = 0; i < max.length(); i++) {
        	rv[i] = max.charAt(i) - '0';
        }
        return rv;
        
    }
    
    public String[] fillDP (String s, int k) {
        String[][] dp = new String[s.length() + 1][Math.min(k, s.length()) + 1];
        for (int i = 0; i <= s.length(); i++){
            dp[i][0] = "";
        }
        for (int j = 1; j <= Math.min(k, s.length()); ++j) {
        	for (int i = j; i <= s.length(); ++i) {
        		if (i == j) {
        			dp[i][j] = s.substring(0, i);
        			continue;
        		}
        		String prev = dp[i - 1][j];
        		String append = dp[i - 1][j - 1] + s.charAt(i - 1);
        		dp[i][j] = append.compareTo(prev) > 0? append: prev;
        	}
        }
        for (int i = 0; i <= s.length(); i++) {
        	System.out.println(Arrays.toString(dp[i]));
        }
        return dp[s.length()];
        
    }
    
    public int[] maxArray(int[] nums, int k) {
        int n = nums.length;
        int[] ans = new int[k];
        for (int i = 0, j = 0; i < n; ++i) {
            while (n - i + j > k && j > 0 && ans[j - 1] < nums[i]) j--;
            if (j < k) ans[j++] = nums[i];
        }
        return ans;
    }
    
    public int countRangeSum(int[] nums, int lower, int upper) {
        long[] rangeSum = new long[nums.length + 1];
        for (int i = 1; i <= nums.length; ++i) {
        	rangeSum[i] = (long)nums[i - 1]+ rangeSum[i - 1];
        }
        long[] copy = Arrays.copyOf(rangeSum, rangeSum.length);
        Arrays.sort(copy);
        HashMap<Long, Integer> argSort = new HashMap<>();
        for (int i = 0; i < copy.length; i++) {
        	argSort.put(copy[i], i);
        }
        //System.out.println(argSort);
        int[] tree = new int[copy.length + 1];
        int sum = 0;
        for (int j = 0; j < rangeSum.length; j++) {
        	long lower_bound = rangeSum[j] - upper;
        	int lb = 0;
        	int ub = copy.length - 1;
        	while (lb <= ub) {
        		int mid = lb + (ub - lb) / 2;
        		if (copy[mid] >= lower_bound)	ub = mid - 1;
        		else	lb = mid + 1;
        	}
        	int from = ub + 1;
        	if (from >= copy.length) {
        		add(tree, argSort.get(rangeSum[j]));
        		continue;
        	}
        	
        	
        	lb = 0;
        	ub = copy.length - 1;
        	
        	long upper_bound = rangeSum[j] - lower;
        	while (lb <= ub) {
        		int mid = lb + (ub - lb) / 2;
        		if (copy[mid] <= upper_bound)	lb = mid + 1;
        		else 	ub = mid - 1;
        	}
        	int to = lb - 1;
        	if (to < 0)	 {
        		add(tree, argSort.get(rangeSum[j]));
        		continue;
        	}
        	System.out.println("from: " + from + " to: " + to);
        	sum += (sum(tree, to) - sum(tree, from - 1));
        	add(tree, argSort.get(rangeSum[j]));
        }
        
        //System.out.println(Arrays.toString(tree));
        return sum;
        
    }
    
    public void add(int[] tree, int pos) {
    	pos += 1;
    	while (pos < tree.length) {
    		tree[pos] += 1;
    		pos += (pos & -pos);
    	}
    }
    
    public int sum (int[] tree, int pos) {
    	//System.out.println("pos = " + pos);
    	pos += 1;
    	int sum = 0;
    	while (pos > 0) {
    		sum += tree[pos];
    		pos -= (pos & -pos);
    	}
    	return sum;
    }
	
	public static void main (String[] args) {
		Solution_301_to_350 s = new Solution_301_to_350();
		int[][] sample = new int[][]{{3,0,1,4,2},{5,6,3,2,1},{1,2,0,1,5},{4,1,0,1,7},{1,0,3,0,5}};
		int[] prices = new int[]{2,1,7,8,0,1,7,3,5,8,9,0,0,7,0,2,2,7,3,5,5};
		int[] a1 = new int[] {2,1,7,8,0,1,7,3,5,8,9,0,0,7,0,2,2,7,3,5,5};
		int[] a2 = new int[] {2,6,2,0,1,0,5,4,5,5,3,3,3,4};
		String[][] tickets = new String[][]{{"EZE","AXA"},{"TIA","ANU"},
			{"ANU","JFK"},{"JFK","ANU"},
			{"ANU","EZE"},{"TIA","ANU"},{"AXA","TIA"},{"TIA","JFK"},{"ANU","TIA"},{"JFK","TIA"}};
		System.out.println(s.countRangeSum(new int[] {-2147483647,0,-2147483647,2147483647},
				-564,
				3864));
	}
	
	

}
