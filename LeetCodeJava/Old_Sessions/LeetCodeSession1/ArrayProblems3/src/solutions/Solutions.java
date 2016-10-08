package solutions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.TreeSet;

public class Solutions {
	
	public long getSumNI(int n, int base) {
		if (n == 1)	return base + 1;
		long half = getSumNI(n / 2, base);
		return half + half * ((long)Math.pow(base, n - n / 2)) - (n % 2 == 0? (long)Math.pow(base, n / 2): 0);
	}
	
	public void recoverArray(String[] strs, int[] nums) {
		for (int i = 0; i < nums.length; ++i) {
			int j = i;
			while (nums[j] != i) {
				swap(strs, nums[j], j);
				int temp = nums[j];
				nums[j] = j;
				j = temp;
			}
			nums[j] = j;
		}
	}
	
	public void swap(String[] strs, int i, int j){
		String s1 = strs[i];
		strs[i] = strs[j];
		strs[j] = s1;
	}
	
	public long fillMatrix2 (int dimension) {
		long[][] dp = new long[dimension + 1][dimension + 1];
		dp[0][0] = 1;
		for (int i = 0; i <= dimension; ++i){
			for (int j = 0; j <= dimension; ++j) {
				if (i == 0 && j == 0)	dp[i][j] = 1;
				else {
					dp[i][j] = (i == 0? 0: dp[i - 1][j]) + (j == 0? 0: dp[i][j - 1]);
				}
			}
		}
		return dp[dimension][dimension];
	}
	
	public long fillMatrix(int dimension) {
		HashMap<Integer, Long> dp = new HashMap<> ();
		dp.put(0, (long) 1);
		int[] array = new int[dimension];
		for (int total = 1; total <= dimension * dimension; ++total){
			fillTable(0, new int[dimension], total, dp, 0);
		}
		Arrays.fill(array, dimension);
		return dp.get(hash(array));
	}
	
	public void fillTable(int pos, int[] array, int total, HashMap<Integer, Long> dp, int prevTotal) {
		if (pos == array.length) {
			long sum = 0;
			for (int i = 0; i < array.length; ++i) {
				if (array[i] == 0)	break;
				if (i == array.length - 1 || array[i] > array[i + 1]) {
					--array[i];
					sum += dp.get(hash(array));
					++array[i];
				}
			}
			dp.put(hash(array), sum);
		}
		else {
			for (int possible = (int) Math.ceil(((double)total - (double)prevTotal) / (array.length - pos));
					possible <= Math.min(pos == 0? Integer.MAX_VALUE: array[pos - 1],
							Math.min(array.length, total - prevTotal)); possible++){
				array[pos] = possible;
				fillTable(pos + 1, array, total, dp, prevTotal + possible);
			}
		}
	}
	
	public int hash(int[] array) {
		int res = 0;
		for (int num: array) {
			res = (res << 3) ^ num;
		}
		return res;
	}
	
	int[] roots;
	int[] rank;
	int islands;
	int[][] board;
	public List<Integer> numIslands2(int m, int n, int[][] positions) {
		int[] dx = new int[] {-1, 1, 0, 0};
		int[] dy = new int[] {0, 0, -1, 1};
        roots = new int[m * n];
        board = new int[m][n];
        rank = new int[m * n];
        islands = 0;
        List<Integer> ret = new ArrayList<Integer>();
        for (int i = 0; i < m * n; i++) {
        	roots[i] = i;
        	rank[i] = 1;
        }
        for (int[] position: positions) {
        	int x = position[0];
        	int y = position[1];
        	islands++;
        	board[x][y] = 1;
        	for (int direc = 0; direc < 4; direc++) {
        		int nx = x + dx[direc];
        		int ny = y + dy[direc];
        		if (nx >=0 && nx < m && ny >= 0 && ny < n && board[nx][ny] == 1)
        			union(x, y, nx, ny);
        	}
        	ret.add(islands);
        }
        return ret;
    }
	
	public int lengthOfLongestSubstringKDistinct(String s, int k) {
		int count = 0;
		int[] c_map = new int[128];
		int i = 0, j = 0, max = 0;
		while (j < s.length()) {
			c_map[s.charAt(j)]++;
			if (c_map[s.charAt(j++)] == 1) {
				count++;
				while (count > k) {
					c_map[s.charAt(i++)]--;
					if (c_map[s.charAt(i - 1)] == 0) {
						count--;
					}
				}
			}
			max = Math.max(max, j - i);
		}
		return max;
    }
	
	public int strobogrammaticInRange(String low, String high) {
        HashMap<Integer, List<String>> sgnums = new HashMap<>();
        ArrayList<String> list = new ArrayList<>();
        list.add("");
        sgnums.put(0, list);
        int count = 0;
        ArrayList<String> list1 = new ArrayList<>();
        list1.add("0");
        list1.add("1");
        list1.add("8");
        if (isValid(low, high, "0"))
        	count++;
        if (isValid(low, high, "1"))
        	count++;
        if (isValid(low, high, "8"))
        	count++;
        sgnums.put(1, list1);
        for (int digit = 2; digit <= high.length(); digit++) {
        	List<String> core = sgnums.get(digit - 2);
        	List<String> thisLength = new ArrayList<>();
        	for (String str: core) {
        		String mod = "8" + str + "8";
        		if (isValid(low, high, mod))
        			count++;
        		thisLength.add(mod);
        		mod = "6" + str + "9";
        		if (isValid(low, high, mod))
        			count++;
        		thisLength.add(mod);
        		mod = "9" + str + "6";
        		if (isValid(low, high, mod))
        			count++;
        		thisLength.add(mod);
        		mod = "1" + str + "1";
        		if (isValid(low, high, mod))
        			count++;
        		thisLength.add(mod);
        		mod = "0" + str + "0";
        		thisLength.add(mod);
        	}
        	sgnums.put(digit, thisLength);
        }
        return count;
        
    }
	private int compareNum(String num1, String num2) {
		if (num1.length() != num2.length()) {
			return num1.length() - num2.length();
		}
		return num1.compareTo(num2);
	}
	
	private boolean isValid (String low, String high, String val) {
		return (compareNum(val, low) >=0 && compareNum(val, high) <= 0);
	}
	
	
	
	public int maxSubArrayLen(int[] nums, int k) {
        int[] sum = new int[nums.length + 1];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 1; i <= nums.length; i++) {
            sum[i] = sum[i - 1] + nums[i - 1];
            map.put(sum[i], i);
        }
        int max = 0;
        for (int i = 0; i <= nums.length; i++) {
            int rem = k + sum[i];
            if (map.containsKey(rem)) {
                max = Math.max(map.get(rem) - i, max);
            }
        }
        return max;
    }
	
	public List<String> findMissingRanges(int[] nums, int lower, int upper) {
		List<String> list = new ArrayList<> ();
        for (int num : nums) {
        	if (num < lower) {
        		continue;
        	}
        	if (num > lower + 1) {
        		list.add(String.valueOf(lower) + "->" + String.valueOf(num - 1));
        	}
        	else if (num == lower + 1) {
        		list.add(String.valueOf(lower));
        	}
        	lower = num + 1;
        }
        if (lower == upper) {
        	list.add(String.valueOf(lower));
        }
        else if (lower < upper) {
        	list.add(lower + "->" + upper);
        }
        return list;
    }
	
	public int shortestWordDistance(String[] words, String word1, String word2) {
		int max = words.length;
		if (word1.equals(word2)) {
			int prev = -words.length;
			for (int i = 0; i < words.length; i++) {
				if (words[i].equals(word1)) {
					max = Math.min(max, i - prev);
					prev = i;
				}
			}
		}
		else {
			int idx1 = -words.length;
			int idx2 = words.length;
			for (int i = 0; i < words.length; i++) {
				if (words[i].equals(word1)) {
					idx1 = i;
					max = Math.min(max, Math.abs(idx1 - idx2));
				}
				if (words[i].equals(word2)) {
					idx2 = i;
					max = Math.min(max, Math.abs(idx1 - idx2));
				}
			}
		}
		return max;
    }
	
	public int minArea(char[][] image, int x, int y) {
        int xMin = image.length;
        int yMin = image[0].length;
        int xMax = 0;
        int yMax = 0;
        for (int i = 0; i < image.length; i++) {
        	for (int j = 0; j < image[0].length; j++) {
        		if (image[i][j] == '1') {
        			xMin = Math.min(xMin, i);
        			xMax = Math.max(xMax, i);
        			yMin = Math.min(yMin, j);
        			yMax = Math.max(yMax, j);
        		}
        	}
        }
        return (xMax - xMin + 1) * (yMax - yMin + 1);
    }
	
	public int threeSumSmaller(int[] nums, int target) {
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < nums.length - 2; i++) {
        	int left = i + 1;
        	int right = nums.length - 1;
        	int rem = target - nums[i];
        	while (right > left) {
        		if (nums[left] + nums[right] >= rem)
        			right--;
        		else {
        			count += (right - left);
        			left++;
        		}
        	}
        }
        return count;
    }
	
	public List<String> generateAbbreviations(String word) {
		List<String> ret = new ArrayList<>();
		ret.add(word);
        if (word.length() == 0) {
        	return ret;
        }
        for (int i = 1; i < word.length(); i++) {
        	//replace the 0---i characters
      	
        	//do not replace the 0---i characters
        	for (String str: generateAbbreviations(word.substring(i))) {
        		ret.add(word.substring(0, i) + str);
        	}
        	
        }
    }
	
	public int bisect_right(int[] nums, int lb, int ub, int target) {
		//looking for the largest element that's less than target
		while (ub >= lb) {
			int mid = lb + (ub - lb) / 2;
			if (nums[mid] < target) {
				lb = mid + 1;
			}
			else {
				ub = mid - 1;
			}
		}
		return lb - 1;
	}
	
	public void wiggleSort(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
        	if (i % 2 == 0 && nums[i + 1] < nums[i]) {
        		int temp = nums[i + 1];
        		nums[i + 1] = nums[i];
        		nums[i] = temp;
        	}
        	else if (i % 2 == 1 && nums[i + 1] > nums[i]) {
        		int temp = nums[i + 1];
        		nums[i + 1] = nums[i];
        		nums[i] = temp;
        	}
        }
    }
	
	public boolean canPermutePalindrome(String s) {
        int[] chars = new int[128];
        for (int i = 0; i < s.length(); i++) {
        	chars[s.charAt(i)]++;
        }
        int odd = 0;
        for (int i = 0; i < 128; i++) {
        	if (chars[i] % 2 == 1)
        		odd++;
        }
        return odd < 2;
    }
	
	public int[][] multiply(int[][] A, int[][] B) {
        int[][] ret = new int[A.length][B[0].length];
        //a map contains all the rows of matrix A
        HashMap<Integer, HashMap<Integer, Integer>> A_rowMap = new HashMap<>();
        for (int i = 0; i < A.length; i++) {
        	HashMap<Integer, Integer> a_row = new HashMap<>();
        	for (int j = 0; j < A[i].length; j++) {
        		if (A[i][j] != 0) {
        			a_row.put(j, A[i][j]);
        		}
        	}
        	if (a_row.size() != 0)
        		A_rowMap.put(i, a_row);
        }
        HashMap<Integer, HashMap<Integer, Integer>> B_colMap = new HashMap<>();
        for (int j = 0; j < B[0].length; j++) {
        	HashMap<Integer, Integer> a_col = new HashMap<>();
        	for (int i = 0; i < B.length; i++) {
        		if (B[i][j] != 0) {
        			a_col.put(i, B[i][j]);
        		}
        	}
        	if (a_col.size() != 0)
        		B_colMap.put(j, a_col);
        }
        //for each element, compute the value
        for (int i = 0; i < ret.length; i++) {
        	if (!A_rowMap.containsKey(i))
        			continue;
        	for (int j = 0; j < ret[0].length; j++) {
        		if (!B_colMap.containsKey(j)) 
        			continue;
        		//so this is the i th row from A and j th col from B
        		HashMap<Integer, Integer> row = A_rowMap.get(i);
        		HashMap<Integer, Integer> col = B_colMap.get(j);
        		for (int colN: row.keySet()) {
        			if (col.containsKey(colN)) {
        				ret[i][j] += (row.get(colN) * col.get(colN));
        			}
        		}
        	}
        }
        return ret;
    }
	
	public boolean canWin(String s) {
		//this hash map basicly means whether the person takes the lead
		//can win the board
        HashMap<String, Boolean> map = new HashMap<>();
        char[] s_arr = s.toCharArray();
        for (int i = 0; i < s_arr.length - 1; i++) {
        	if (s_arr[i] == '+' && s_arr[i + 1] == '+' && dfs(s_arr, i, map)) {
        		return true;
        	}
        }
        return false;
    }
	
	//this function returns whether the person can win by fliping the two ++ at start
	public boolean dfs (char[] s, int start, HashMap<String, Boolean> map) {
		s[start] = '-';
		s[start + 1] = '-';
		//default is he can win, since is possible that no flip is allowed
		boolean canWin = true;
		//if the next can win, then we can not, vice versa
		if (map.containsKey(String.valueOf(s))) {
			canWin = !map.get(String.valueOf(s));
		}
		else {
			//search for every possibility
			for (int i = 0; i < s.length - 1; i++) {
				//if we found the next and by flipping that the next person
				//is guranteed to win, we know we are doomed to fail
				if (s[i] == '+' && s[i + 1] == '+' && dfs(s, i, map)) {
					canWin = false;
					break;
	        	}
			}
		}
		//so what we want to do is to put the descendant to be the reverse of can win
		map.put(String.valueOf(s), !canWin);
		s[start] = '+';
		s[start + 1] = '+';
		return canWin;
	}
	
	public void union (int x1, int y1, int x2, int y2) {
		int pos1 = x1*board[0].length + y1;
		int pos2 = x2*board[0].length + y2;
		while (roots[pos1] != pos1) {
			pos1 = roots[pos1];
		}
		while (roots[pos2] != pos2) {
			pos2 = roots[pos2];
		}
		if (pos1 == pos2)
			return;
		islands--;
		if (rank[pos1] > rank[pos2]) {
			roots[pos2] = pos1;
		}
		else if (rank[pos1] < rank[pos2]) {
			roots[pos1] = pos2;
		}
		else {
			roots[pos2] = pos1;
			rank[pos1]++;
		}
	}

	
	public boolean isReflected(int[][] points) {
		if (points==null || points.length == 0)
			return false;
		HashMap<Integer, HashMap<Integer, Integer>> points_map = new HashMap<>();
		int minX = 0;
		int maxX = 0;
		for (int[] point: points) {
			int x = point[0];
			maxX = Math.max(x, maxX);
			minX = Math.min(x, minX);
			int y = point[1];
			if (!points_map.containsKey(x)) {
				HashMap<Integer, Integer> map = new HashMap<>();
				map.put(y, 1);
				points_map.put(x, map);
			}
			else {
				HashMap<Integer, Integer> map = points_map.get(x);
				if (map.containsKey(y)) {
					map.put(y, map.get(y) + 1);
				}
				else {
					map.put(y, 1);
				}
			}
		}
		for (int x: points_map.keySet()) {
			if (x - minX == maxX - x)
				continue;
			HashMap<Integer, Integer> map = points_map.get(x);
			//get the reverse one
			int reflectedX = maxX - (x - minX);
			if (!points_map.containsKey(reflectedX))
				return false;
			HashMap<Integer, Integer> reflectedmap = points_map.get(reflectedX);
			for (int y:map.keySet()) {
				if (!reflectedmap.containsKey(y) || reflectedmap.get(y) != map.get(y))
					return false;
			}
		}
		return true;
		
    }
	
	public int maxKilledEnemies(char[][] grid) {
		int n = grid.length;
		int m = grid[0].length;
		
        int[][] maxRows = new int[n][m];
        for (int i = 0; i < n; i++) {
        	int start = 0;
        	while (start < m) {
        		if (grid[i][start] == 'W') {
        			start++;
        			continue;
        		}
        		int end = start;
        		int enemies = 0;
        		while (end <= m) {
        			if (end == m || grid[i][end] == 'W') {
        				for (int idx = start; idx < end; idx++) {
        					if (grid[i][idx] != 'E')
        						maxRows[i][idx] = enemies;
        				}
        				start = end + 1;
        				break;
        			}
        			else if (grid[i][end] == 'E') {
        				enemies++;
        			}
        			end++;
        		}
        	}
        }
        
        int max = 0;
        for (int j = 0; j < m; j++) {
        	int start = 0;
        	while (start < n) {
        		if (grid[start][j] == 'W') {
        			start++;
        			continue;
        		}
        		int end = start;
        		int enemies = 0;
        		while (end <= n) {
        			if (end == n || grid[end][j] == 'W') {
        				for (int idx = start; idx < end; idx++) {
        					if (grid[idx][j] != 'E')
        						max = Math.max(maxRows[idx][j] + enemies, max);
        				}
        				start = end + 1;
        				break;
        			}
        			else if (grid[end][j] == 'E') {
        				enemies++;
        			}
        			end++;
        		}
        	}
        }
        for (int i = 0; i < n; i++) {
        	System.out.println(Arrays.toString(maxRows[i]));
        }
        return max;
    }

	
	public String rearrangeString(String str, int k) {
		if (str == null || str.length() == 0 || k <= 1)
			return "";
        int n = str.length();
        HashMap<Character, Integer> counts = new HashMap<>();
        PriorityQueue<charIdxPair> pq = new PriorityQueue<>();
        TreeSet<Integer> idxes = new TreeSet<>();
        for (int i= 0; i < str.length(); i++) {
        	int count = counts.containsKey(str.charAt(i))? counts.get(str.charAt(i)): 0;
        	counts.put(str.charAt(i), count + 1);
        	idxes.add(i);
        }
        
        for (Character c : counts.keySet()) {
        	pq.offer(new charIdxPair(c, counts.get(c)));
        }
        char[] ret = new char[n];
        while (!pq.isEmpty()) {
        	charIdxPair pair = pq.poll();
        	int start = Integer.MIN_VALUE;
        	for (int i = 0; i < pair.i; i++) {
        		if (idxes.ceiling(start + k) == null)
        			return "";
        		start = idxes.ceiling(start + k);
        		System.out.println(start);
        		ret[start] = pair.c;
        		idxes.remove(new Integer(start));
        	}
        }
        return String.valueOf(ret);
    }
	
	class charIdxPair implements Comparable<charIdxPair>{
		char c;
		int i;
		charIdxPair (char c, int idx) {
			this.c = c;
			this.i = idx;
		}
		@Override
		public int compareTo(charIdxPair o) {
			// TODO Auto-generated method stub
			return o.i - i;
		}
		
		
	}
	
	
	public int maxSumSubmatrix(int[][] matrix, int k) {
        int n = matrix.length;
        int m = matrix[0].length;
        int[][] aux = new int[n + 1][m];
        for (int i = 1; i <= n; i++) {
        	for (int j = 0; j < m; j++) {
        		aux[i][j] = aux[i - 1][j] + matrix[i - 1][j];
        	}
        }
        int opt = -10000;
        for (int i = 1; i <= n; i++) {
        	for(int j = 0; j < i; j++) {
        		TreeSet<Integer> set = new TreeSet<>();
        		set.add(0);
        		int sum = 0;
        		for (int l = 0; l < m; l++) {
        			sum += (aux[i][l] - aux[j][l]);
        			//in order for it to find sum - x <= k
        			//that x >= sum - k
        			int find = sum - k;
        			

        			Integer ceil = set.ceiling(find);
        			if (ceil != null && (k + ceil - sum) < k - opt) {
        				opt = sum - ceil;
        			}
        			set.add(sum);
        		}
        	}
        }
        return opt;
    }
	
	class indSum {
		int x;
		int y;
		int sum;
		indSum(int x, int y, int sum) {
			this.x = x;
			this.y = y;
			this.sum = sum;
		}
	}
	
	public static void main (String[] args) {
		Solutions s = new Solutions();
		String[] strs = new String[]{"cat", "mouse", "dog", "rabbit", "tiger", "lion"};
		int[] shifts = new int[]{2, 0, 1, 3, 5, 4};
		s.recoverArray(strs, shifts);
		System.out.println(s.getSumNI(4, 4));
	}
	
}
