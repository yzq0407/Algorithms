import java.util.*;

public class Solution_51_to_100 {
	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> list = new ArrayList<>();
		list.add(new ArrayList<Integer>());
		if (nums== null) {
			return list;
		}
		dfs(list, nums, 0);
		return list;
    }
	
	public int numTrees(int n) {
        if (n == 0) {
            return 1;
        }
        if (n < 3) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; ++i) {
            for (int root = 1; root <= i; root++) {
                int left = dp[root - 1];
                int right = dp[i - root];
                dp[i] += (left * right);
            }
        }
        return dp[n];
    }
	
	public int maximalRectangle(char[][] matrix) {
		int n = matrix.length;
		int m = matrix[0].length;
		int[][] cumulativeSum = new int[n][m];
		for (int j = 0; j < m; ++j) {
			for (int i = 0; i < n; ++i) {
				if (matrix[i][j] == '0') {
					cumulativeSum[i][j] = 0;
				}
				else {
					int up = (i == 0)? 0: cumulativeSum[i - 1][j];
					cumulativeSum[i][j] = up + 1;
				}
			}
		}
		int max = 0;
		for (int i = 0; i < n; i++) {
			max = Math.max(max, largestRectangleArea(cumulativeSum[i]));
		}
		return max;
    }
	
	public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        for (int i = 0; i <= heights.length; ++i) {
            while (!stack.isEmpty() && (i == heights.length ||heights[stack.peek()] >= heights[i])) {
                int height = heights[stack.pop()];
                int left = stack.isEmpty()? 0: stack.peek() + 1;
                int right = i;
                maxArea = Math.max((right - left) * height, maxArea);
            }
            stack.push(i);
        }
        return maxArea;
    }
	
	//82
	public ListNode deleteDuplicates(ListNode head) {
		if (head == null)
			return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode last = dummy;
        ListNode current = head;
        while (current != null) {
        	while (current.next != null && current.next.val == current.val) {
        		int skip = current.val;
        		while (current != null && current.val == skip) {
        			current = current.next;
        		}
        		if (current == null)
        			break;
        	}
        	last.next = current;
        	last = current;
        	if (current != null) {
        		current = current.next;
        	}
        }
        return dummy.next;
        
    }
	
	public void dfs (List<List<Integer>> list, int[] nums, int start) {
		if (start == nums.length - 1) {
			ArrayList<Integer> sublist = new ArrayList<>();
			sublist.add(nums[start]);
			list.add(sublist);
			return;
		}
		dfs (list, nums, start + 1);
		for (int i = 0; i < list.size(); i++) {
			List<Integer> sublist = list.get(i);
			ArrayList<Integer> newlist = new ArrayList<>();
			newlist.addAll(sublist);
			newlist.add(nums[start]);
			list.add(newlist);
		}
	}
	
	public List<Interval> merge(List<Interval> intervals) {
        List<EntryPoint> list = new ArrayList<>();
        for (Interval interval: intervals) {
            list.add(new EntryPoint(interval.start, true));
            list.add(new EntryPoint(interval.end, false));
        }
        Collections.sort(list);
        List<Interval> rv = new ArrayList<> ();
        int layer = 0;
        int start = 0;
        for (EntryPoint ep: list) {
            if (ep.isStart) {
                if (layer == 0)
                    start = ep.val;
                layer++;
            }
            else {
                layer--;
                if (layer == 0) {
                    rv.add(new Interval(start, ep.val));
                }
            }
        }
        return rv;
    }
    
    public class EntryPoint implements Comparable<EntryPoint> {
        int val;
        boolean isStart;
        
        public EntryPoint(int val, boolean isStart) {
            this.val = val;
            this.isStart = isStart;
        }
        
        public int compareTo(EntryPoint ep) {
            if (val != ep.val)
                return val - ep.val;
            else if (isStart){
                return -1;
            }
            return 1;
            
        }
    }
    
    public List<String> restoreIpAddresses(String s) {
        List<String> rv = new ArrayList<>();
        dfs(rv, s, 0, 0, "");
        return rv;
    }
    
    public void dfs(List<String> rv, String s, int pos, int segs, String sofar) {
        if (segs == 3 && Long.parseLong(s.substring(pos)) < 256 && (pos == s.length() - 1 || s.charAt(pos) != '0')){
            rv.add(sofar + s.substring(pos));
            return;
        }
        else if (segs == 3 || (segs < 3 && pos >= s.length())) {
            return;
        }
        for (int i = pos + 1; i < Math.min(pos + 4, s.length()); i++){
            int segment = Integer.parseInt(s.substring(pos, i));
            if (segment < 256 && (s.charAt(pos) != '0' || i > pos + 1)){
                dfs(rv, s, i, segs + 1, sofar + s.substring(pos, i) + ",");
            }
        }
    }
    
    public List<Integer> grayCode(int n) {
        boolean[] isVisited = new boolean[1<<n];
        int[] gray = new int[1<<n];
        dfs(gray, isVisited, 0, n);
        List<Integer> rv = new ArrayList<>();
        for (int code: gray) {
            rv.add(code);
        }
        return rv;
        
    }
    
    public boolean dfs(int[] gray, boolean[] isVisited, int pos, int n) {
        if (pos == gray.length - 1) {
            return true;
        }
        isVisited[gray[pos]] = true;
        for (int i = 0; i < n; i++) {
            int neighbor = gray[pos] ^ (1 << i);
            if (!isVisited[neighbor]) {
                gray[pos + 1] = neighbor;
                if (dfs(gray, isVisited, pos + 1, n))
                    return true;
            }
        }
        isVisited[gray[pos]] = false;
        return false;
    }
    
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s3.length() != s1.length() + s2.length()) {
            return false;
        }
        boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
        dp[0][0] = true;
        //dp[m][n] --> s2[0:m + n] intervened by s1[0:m] and s2[0:n]
        //dp[m][n] = (dp[m -1][n] && s1.charAt(m) == s3.charAt(m + n)) ||(...)
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i == 0 && j == 0) {
                    continue;
                }
                if (i == 0) {
                    dp[0][j] = dp[0][j - 1] && (s2.charAt(j - 1) == s3.charAt(j - 1));
                }
                else if (j == 0) {
                    dp[i][0] = dp[i - 1][0] && (s1.charAt(i - 1) == s3.charAt(i - 1));
                }
                else {
                    dp[i][j] = (dp[i - 1][j] && (s1.charAt(i - 1) == s3.charAt(i + j - 1)))
                    || (dp[i][j - 1] && (s2.charAt(j - 1) == s3.charAt(i + j - 1)));
                }
            }
        }
        for(int i = 0; i < s1.length() + 1; i++) {
        	System.out.println(Arrays.toString(dp[i]));
        }
        return dp[s1.length()][s2.length()];
    }
    
    public int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        int count = 1;
        for (int offset = 0; 2 * offset < n; offset++) {
            //i = offset j = offset -- n - offset - 1
            for (int j = offset; j < n - offset - 1; j++) {
                matrix[offset][j] = count++;
            }
            for (int i = offset; i < n - offset - 1; i++) {
                matrix[i][n - offset - 1] = count++;
            }
            for (int j = n - offset - 1; j > offset; j--) {
                matrix[n - offset - 1][j] = count++;
            }
            for (int i = n - offset - 1; i > offset; i--) {
                matrix[i][offset] = count++;
            }
        }
        if (n % 2 == 1) {
            matrix[n / 2][n / 2] = count;
        }
//        for (int i = 0; i < n; i++) {
//        	System.out.println(Arrays.toString(matrix[i]));
//        }
        return matrix;
    }
    
    public int minPathSum(int[][] grid) {
        if (grid.length == 0) {
            return 0;
        }
        for (int j = 1; j < grid[0].length; j++) {
            grid[0][j] += grid[0][j - 1];
        }
        for (int i = 1; i < grid.length; i++) {
            grid[i][0] += grid[i - 1][0];
            for (int j = 1; j < grid[0].length; ++j) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            } 
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }
    
    public int minDistance(String word1, String word2) {
        //if (word1.length() == 0 || word2.length() == 0) return Math.max(word1.length(), word2.length());
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        //dp[i][j] represents the edit distance between word1[0:i] and word2[0:j]
        //fill out base case where we only have one character
        for (int i = 0; i <= word1.length(); i++) {
            for (int j = 0; j <= word2.length(); j++) {
                //three cases:
                //1. match end character dp[i][j] = dp[i - 1][j - 1];
                //2. match word1[0:i] and word2[0:j -1] and add word2[j]
                //3. match word2[0:j] and word1[0:i - 1] and add word1[i]
                if (i == 0) {
                    dp[0][j] = j;
                    continue;
                }
                if (j == 0) {
                    dp[i][0] = i;
                    continue;
                }
                int match = word1.charAt(i - 1) == word2.charAt(j - 1)? dp[i- 1][j - 1]: dp[i - 1][j - 1] + 1;
                dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]) + 1, match);
            }
        }
        return dp[word1.length()][word2.length()];
    }
    
    public String getPermutation(int n, int k) {
        int multi = 1;
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            list.add(i);
            if (i != n)
                multi *= i;
        }
        StringBuilder sb = new StringBuilder();
        k = k - 1;
        while (!list.isEmpty()) {
            int pos = k / multi;
            sb.append(list.remove(pos));
            if (list.isEmpty())
            	break;
            k %= multi;
            multi /= (n - 1);
            n--;
        }
        return sb.toString();
    }
    
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> list = new ArrayList<> ();
        int n = matrix.length;
        if (n == 0) return list;
        int m = matrix[0].length;
        for (int offset = 0; offset + offset <= Math.min(m, n); offset++) {
            for (int i = offset; i < m - offset - 1; i++) list.add(matrix[offset][i]);
            for (int i = offset; i < n - offset - 1; i++) list.add(matrix[i][m - offset - 1]);
            for (int i = m - offset - 1; i > offset; i--) list.add(matrix[n - offset - 1][i]);
            for (int i = n - offset - 1; i > offset; i--) list.add(matrix[i][offset]);
        }
        if (n == m && n % 2 == 1) list.add(matrix[n / 2][n / 2]);
        return list;
    }
    
    
    public boolean isScramble(String s1, String s2) {
        if (s1.length() != s2.length())
            return false;
        int len = s1.length();
        //dp i, j, k == s1(i: i + k) is scramble of s2(j : j + k)
        boolean[][][] dp = new boolean[len][len][len + 1];
        for (int k = 1; k <= len; k++) {
            for (int i = 0; i <= len - k; i++) {
                for (int j = 0; j <= len - k; j++) {
                    if (s1.substring(i, i + k).equals(s2.substring(j, j + k))) {
                    	dp[i][j][k] = true;
                    	continue;
                    }
                    for (int z = 1; z < k; z++) {
                        if ((dp[i][j][z] && dp[i + z][j + z][k - z]) ||
                        (dp[i][j + k - z][z] && dp[i + z][j][k - z])) {
                        	//System.out.println("i: " + i + "  j: " + j + "  k: " + k);
                            dp[i][j][k] = true;
                            break;
                        }
                    }
                }
            }
        }
        return dp[0][0][len];
    }

	public static void main(String[] args) {
		List<Interval> list = new ArrayList<Interval>();
		int[][] test = new int[][]{{1, 3}, {1, 5}, {4, 2}};
		Solution_51_to_100 s = new Solution_51_to_100();
		System.out.println(s.isScramble("rgeat","great"));
	}
}
