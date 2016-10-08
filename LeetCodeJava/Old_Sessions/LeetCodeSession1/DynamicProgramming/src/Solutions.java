import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.Stack;


public class Solutions {
	public boolean isMatch(String s, String p) {
        if (s.length() ==0 && p.length() ==0) 
        	return true;
        if (p.length()>=2 && p.charAt(1) == '*') {
        	if (isMatch(s, p.substring(2)))
        		return true;
        	int i = 0;
        	while (i<s.length() && isMatch(s.charAt(i), p.charAt(0))) {
        		if (isMatch(s.substring(i+1), p.substring(2)))
    				return true;
        		i++;
        	}
        }
        if (s.length() ==0 || p.length() ==0) 
        	return false;
        
        if (p.charAt(0) == '.' && s.length()!=0){
        	if (isMatch (s.substring(1), p.substring(1)))
        		return true;
        }
        if (p.charAt(0) == s.charAt(0)) 
        	return isMatch(s.substring(1), p.substring(1));
        return false;
    }
	
	
	public int minCost(int[][] costs) {
		if (costs.length == 0)
			return 0;
		int n = costs.length;
		int[][] DP = new int[n][3];
		DP[0] = costs[0];
		for (int i = 1; i < n; i++) {
			for (int j = 0; j < 3; j++) {
				DP[i][j] = Math.min(DP[i - 1][(j + 2) % 3], DP[i - 1][(j + 1) % 3]) 
						+ costs[i][j];
			}
		}
		return Math.min(DP[n - 1][0], Math.min(DP[n - 1][1], DP[n - 1][2]));
    }

	public int minCut(String s) {
		int[] cuts = new int[s.length()];
		Arrays.fill(cuts, s.length());
		for (int i = 0; i<s.length(); i++) {
			int p1 = i-1;
			int p2 = i+1;
			while (p1>=0 && p2<s.length() && s.charAt(p1) == s.charAt(p2)) {
				if (p1==0) {
					cuts[p2] = 0;
					break;
				}
				else {
					cuts[p2] = Math.min(cuts[p1-1]+1, cuts[p2]);
					p1--;
					p2++;
				}
			}
			p1 = i-1;
			p2 = i;
			while (p1>=0 && p2<s.length() && s.charAt(p1) == s.charAt(p2)) {
				if (p1==0) {
					cuts[p2] = 0;
					break;
				}
				else {
					cuts[p2] = Math.min(cuts[p1-1]+1, cuts[p2]);
					p1--;
					p2++;
				}
			}
			
		}
		return cuts[s.length()-1];
	}

	
	public boolean isMatch (char s, char p) {
		if (p=='.')
			return true;
		return s==p;
	}
	
	
	public int longestValidParentheses(String s) {
		if (s==null ||s.length()<2)
			return 0;
		boolean[] validPos = new boolean[s.length()];
		Stack<Integer> pos = new Stack<> ();
		for (int i = 0; i<s.length(); i++) {
			if (s.charAt(i) =='(')
				pos.add(i);
			else if (!pos.isEmpty()){
				validPos[i] = true;
				validPos[pos.pop()] = true;
			}
		}
		int max = 0, i = 0;
		while (i<s.length()) {
			int j = i+1;
			if (validPos[i]) {
				while (j<s.length() && validPos[j]) j++;
				max = Math.max(j-i, max);
			}
			i = j;
		}
		return max;
    }
	

	public int calculateMinimumHP(int[][] dungeon) {
        int n = dungeon.length;
        int m = dungeon[0].length;
        int[][] DP = new int[n][m];
        DP[n-1][m-1] = dungeon[n-1][m-1]<0? Math.abs(dungeon[n-1][m-1])+1: 1;
        for (int j = m-2; j>=0; j--) {
        	DP[n-1][j] = DP[n-1][j+1] - dungeon[n-1][j]>0?DP[n-1][j+1] - dungeon[n-1][j]: 1;
        }
        
        for (int i = n-2; i>=0; i--) {      	
        	DP[i][m-1] = DP[i+1][m-1] - dungeon[i][m-1]>0?DP[i+1][m-1] - dungeon[i][m-1]:1;
        }
        for (int i = m-2; i>=0; i--){
        	for (int j = n-2; j>=0; j--){
        		int betterStep = Math.min(DP[i+1][j], DP[i][j+1]);
        		DP[i][j] = betterStep - dungeon[i][j]>0?betterStep - dungeon[i][j]:1;
        	}
        }
        return DP[0][0];
    }
	
	public int jump(int[] nums) {
        int range = 0;
        int jumps = 0;
        int i = 0;
        while (range < nums.length-1) {
        	int thisStep = i; 
        	for (; i<=range; i++) {
        		thisStep = Math.max(nums[i] + i, thisStep); 
        	}
        	range = thisStep;
        	jumps++;
        }
        return jumps;
    }
	
	
	public int[] maxNumber(int[] nums1, int[] nums2, int k) {
		if (nums1==null ||nums1.length==0)
			return nums2;
		if (nums2==null ||nums2.length==0)
			return nums1;
        String[] max1 = maxNumber(nums1, k);
        String[] max2 = maxNumber(nums2, k);
        //this case is for the all from nums1
        String maxNumber = max1[k-1];
        
        for (int i = 0; i<k-1; i++){
        	//two segments, one has length x, the other has length k-x
        	String seg1 = max1[i];
        	String seg2 = max2[k-2-i];
        	int j = 0;
        	int w = 0;
        	StringBuilder merge = new StringBuilder();
        	//try merge the two segments together
        	while (j<seg1.length()&& w<seg2.length()) {
        		if (seg1.charAt(j) >= seg2.charAt(w)){
        			merge.append(seg1.charAt(j++));
        			continue;
        		}
        		if (seg2.charAt(w) > seg1.charAt(j)) {
        			merge.append(seg2.charAt(w++));
        		}
        	}
        	if (j<seg1.length())
        		merge.append(seg1.substring(j));
        	if (w<seg2.length())
        		merge.append(seg2.substring(w));
        	String mergeStr = merge.toString();
        	//we need to look at the length, if one length is longer than the other, it is strictly bigger
        	if (mergeStr.length()!=maxNumber.length()){
        		maxNumber = mergeStr.length()>maxNumber.length()? mergeStr: maxNumber;
        		continue;
        	}
        	maxNumber = maxNumber.compareTo(mergeStr)>=0? maxNumber : mergeStr;
        }
        if (max2[k-1].length()!=maxNumber.length()){
    		maxNumber = max2[k-1].length()>maxNumber.length()? max2[k-1]: maxNumber;
    	}
        else
        	maxNumber = maxNumber.compareTo(max2[k-1])>=0? maxNumber: max2[k-1];
        //turn into the array
        int[] ret = new int[maxNumber.length()];
        for (int i = 0; i<maxNumber.length(); i++) {
        	ret[i] = maxNumber.charAt(i) - '0';
        }
        return ret;
    }
	
	//return the maximum k-digit number using nums in the array following lexi order
	public String[] maxNumber (int[] nums,int k){
		String[] maxArray = new String[k];
		String[] ret = new String[nums.length];
		int max = Integer.MIN_VALUE;
		for (int i = 0;i<nums.length; i++){
			max = Math.max(max, nums[i]);
			ret[i] = String.valueOf(max);
		}
		maxArray[0] = ret[nums.length-1];
		for (int p = 1; p<k; p++){
			String prefix = "";
			String[] aux = new String[nums.length];
			for (int i = 0; i<nums.length; i++){
				if (i<=p) {
					prefix = prefix + nums[i];
					aux[i] = prefix;
					continue;
				}
				String choice = ret[i-1] + String.valueOf(nums[i]);
				aux[i] = aux[i-1].compareTo(choice)>=0? aux[i-1]: choice;
			}
			ret = aux;
			maxArray[p] = ret[nums.length-1];
		}
		return maxArray;
	}
	
	class squareVertex {
		int squareSide = 1;
		int leftSpan = 1;
		int upSpan = 1;
		
		squareVertex(int side, int left, int up){
			squareSide = side;
			leftSpan = left;
			upSpan = up;
		}
		
		public String toString(){
			return "[" + squareSide + "," + leftSpan + "," + upSpan + "]";
		}
	}
	
	public int maximalSquare(char[][] matrix) {
		if (matrix==null || matrix.length==0 ||matrix[0].length==0)
			return 0;
		int n = matrix.length;
		int m = matrix[0].length;
        squareVertex[][] dp = new squareVertex[n][m];
        int max = 0;
        if (matrix[0][0]=='1') {
        	dp[0][0] = new squareVertex(1, 1, 1);
        	max = 1;
        }
        else
        	dp[0][0] = new squareVertex(0, 0, 0);
        for (int j = 1; j<m; j++) {
        	if (matrix[0][j]=='0') {
        		dp[0][j] = new squareVertex(0, 0, 0);
        		continue;
        	}
        	dp[0][j] = new squareVertex(1, dp[0][j-1].leftSpan+1, 1);
        	max = 1;
        }
        for (int i = 1; i<n; i++) {
        	if (matrix[i][0] =='0'){
        		dp[i][0] = new squareVertex(0, 0, 0);
        		continue;
        	}
        	dp[i][0] = new squareVertex(1, 1, dp[i-1][0].upSpan+1);
        	max  =1;
        }
        for (int i=1; i<n; i++) {
        	for (int j=1; j<m; j++){
        		if (matrix[i][j]=='0') {
        			dp[i][j] = new squareVertex(0, 0, 0);
        			continue;
        		}
        		int left = dp[i][j-1].leftSpan +1;
        		int up = dp[i-1][j].upSpan + 1;
        		int areaSide = Math.min(Math.min(dp[i-1][j-1].squareSide+1, left), up);
        		if (areaSide>max)
        			max = areaSide;
        		dp[i][j] = new squareVertex(areaSide, left, up);
        	}
        }
        for (int i = 0; i<n; i++) {
        	for (int j = 0; j<m; j++){
        		System.out.print(dp[i][j] + "    ");
        	}
        	System.out.println("");
        }
        return max*max;
    }
	
	
	public int[] sliceInt (String nStr){
		int length = nStr.length();
		int[] ret = new int[length];
		for (int i= 0; i<length; i++){
			ret[i] = nStr.charAt(i) - '0';
		}
		return ret;
	}
	
	public String getMaxNumber(int[] nums1, int[] nums2, int i, int j){
		StringBuilder sb = new StringBuilder();
		while (i<nums1.length && j<nums2.length){
			if (nums1[i]>=nums2[j]){
				sb.append(nums1[i++]);
			}
			else {
				sb.append(nums2[j++]);
			}
		}
		while (i<nums1.length) {
			sb.append(nums1[i++]);
		}
		while (j<nums2.length){
			sb.append(nums2[j++]);
		}
		return sb.toString();
	}
	
	public int countPrimes(int n) {
		boolean[] isComposite = new boolean[n];
		isComposite[0] = true;
		isComposite[1] = true;
		for (int i = 2; i * i < n ; i++) {
			if (!isComposite[i]) {
				for (int j = i* i ; j < n; j += i) {
					isComposite[j] = true;
				}
			}
		}
		int count = 0;
		for (int i = 2; i < n; i++) {
			if (!isComposite[i])
				count++;
		}
		return count;
    }
	
	public int maximalRectangle(char[][] matrix) {
		if (matrix==null)
			return 0;
		int n = matrix.length;
		int m = matrix[0].length;
		if (n==0 || m==0)
			return 0;
        int[][] dpHz  = new int[n][m];
        int[][] dpVt = new int[n][m];
        int maxArea = 0;
        if (matrix[0][0] == '1') {
        	dpHz[0][0] = 1;
        	dpVt[0][0] = 1;
        	maxArea = 1;
        }
        for (int i = 1; i< n; i++) {
        	if (matrix[i][0] == '1') {
        		dpHz[i][0] = 1;
        		dpVt[i][0] = dpVt[i-1][0] +1;
        		maxArea = Math.max(dpVt[i][0], maxArea);
        	}	
        }
        for (int i = 1; i< m; i++) {
        	if (matrix[0][i] == '1') {
        		dpHz[0][i] = dpHz[0][i-1] +1;
        		dpVt[0][i] = 1;
        		maxArea = Math.max(dpHz[0][i], maxArea);
        	}	
        }
        for (int i = 1; i<n; i++) {
        	for (int j = 1; j<m; j++) {
        		if (matrix[i][j] == '1') {
	        		dpHz[i][j] = Math.min(dpHz[i][j-1], dpHz[i-1][j-1])+1;
	        		dpVt[i][j] = Math.min(dpVt[i-1][j], dpVt[i-1][j-1])+1;
	        		maxArea = Math.max(dpHz[i][j]*dpVt[i][j], maxArea);
        		}
        	}
        }
        printTable(matrix);
        System.out.println("------------------");
        printTable(dpHz);
        System.out.println("------------------");
        printTable(dpVt);
        return maxArea;
    }
	
	public int maxProfit(int k, int[] prices) {
		if (k == 0)
			return 0;
		if (prices.length <= 2 * k) {
			int sum = 0;
			for (int i = 1; i < prices.length; i++) {
				if (prices[i] > prices[i - 1])
					sum += (prices[i] - prices[i - 1]);
			}
			return sum;
		}
		int[] dp = new int[k * 2];
		for (int i = 0; i < k; i ++) {
			dp[2 * i] = Integer.MIN_VALUE;
		}
		for (int price: prices) {
			if (-price > dp[0]) 
				dp[0] = -price;
			for (int i = 1; i < 2 * k; i++) {
				dp[i] = Math.max(dp[i], dp[i - 1] + price * ((i % 2) * 2 - 1));
			}
		}
		return dp[2 * k - 1];
    }
	
	public int largestRectangleArea(int[] heights) {
		if (heights==null || heights.length==0)
			return 0;
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0, i = 0;
        while (i<heights.length ||!stack.isEmpty()) {
        	if (i==heights.length || (!stack.isEmpty() && heights[stack.peek()]>heights[i])){
        		int height = stack.pop();
        		int left = stack.isEmpty()? 0: stack.peek()+1;
        		maxArea = Math.max(heights[height]*(i-left), maxArea);
        	}
        	else {
        		stack.push(i++);
        	}
        }
        return maxArea;
    }
	
	public int coinChange(int[] coins, int amount) {
		int[] coinNumber = new int[amount+1];
		for (int i = 1; i <= amount; i++) {
			coinNumber[i] = -1;
			int min = Integer.MAX_VALUE;
			for (int coin: coins) {
				if (i - coin >= 0 && coinNumber[i - coin] != -1) {
					min = Math.min(min, coinNumber[i - coin] + 1);
				}
			}
			if (min != Integer.MAX_VALUE) {
				coinNumber[i] = min;
			}
		}
		return coinNumber[amount];
    }
	
//	public int maxCoins(int[] nums) {
//        int[][] dp = new int[nums.length][nums.length];
//        for (int i = 0; i < nums.length; i++) {
//        	int left = i == 0? 1: nums[i - 1];
//        	int right = i == nums.length - 1? 1 : nums[i + 1]; 
//        	dp[i][i] = left * nums[i] * right;
//        }
//        for (int range = 2; range <= nums.length; range++) {
//        	for (int i = 0; i <= nums.length - range; i++) {
//        		for (int j = i; j < i + range; j++) {
//        			int left = i == 0? 1: nums[i - 1];
//        			int right = i + range - 1 == nums.length - 1? 1: nums[i + range];
//        			
//        			int leftBurst = j== i? 0: dp[i][j - 1];
//        			int rightBurst = j == i + range - 1? 0: dp[j + 1][i + range - 1];
//        			dp[i][i + range - 1] = Math.max(dp[i][i + range - 1],
//        					leftBurst + left * nums[j] *right + rightBurst);
//        		}
//        	}
//        }
//        return dp[0][nums.length - 1];
//    }
	
	public int maxProfit(int[] prices) {
		if (prices == null || prices.length == 0)
            return 0;
		int[] left_max = new int[prices.length];
		int[] right_max = new int[prices.length];
		//left_max[i] means the maximum profit we can make by considering 0 ---- i day
		//right_max[i] means the maximum profit we can make by considering i --- prices,length-1 day
		//we will just use the same algorithm
		left_max[0] = 0;
		int left_min = prices[0];
		for (int i = 1; i < prices.length; i++) {
			left_max[i] = Math.max(left_max[i-1], prices[i] - left_min);
			left_min = Math.min(left_min, prices[i]);
		}
		right_max[prices.length - 1] = 0;
		int right_maxim = prices[prices.length - 1];
		for (int i = prices.length -2; i >= 0; i--) {
			right_max[i] = Math.max(right_max[i+1], right_maxim - prices[i]);
			right_maxim = Math.max(right_maxim, prices[i]);
		}
		int max = left_max[prices.length - 1];
		for (int i = 0; i < prices.length - 1; i++) {
			max = Math.max(max, left_max[i] + right_max[i + 1]);
		}
		return max;
    }
	
	public int maxCoins(int[] nums) {
        if (nums==null || nums.length==0)
            return 0;
        int n = nums.length;
        int[][] dp = new int[n][n];
        for (int i = 0; i<n; i++)
        	dp[i][i] = nums[i];
    	for (int offset = 1; offset < n; offset++){
    		for (int i = 0; i + offset<n; i++) {
    			int j = i+offset;
    			for (int x = i; x<=j; x++) {
    				if (x == i)
    					dp[i][j] = Math.max(dp[i][j], dp[x+1][j] + calculateCoint(nums, x, i-1, j+1));
    				else if (x==j)
    					dp[i][j] = Math.max(dp[i][j], dp[i][x-1] + calculateCoint(nums, x, i-1, j+1));
    				else	
    					dp[i][j] = Math.max(dp[i][j], dp[i][x-1] + calculateCoint (nums, x, i-1, j+1) + dp[x+1][j]);
    			}
    		}
    	}
    	for (int i = 0; i < n; i++) {
    		for (int j = 0; j< n; j++) {
    			System.out.print (dp[i][j] + " ");
    		}
    		System.out.println("");
    	}
    	return dp[0][n-1];
    }
	
	public boolean isInterleave(String s1, String s2, String s3) {
		int l1 = s1.length();
		int l2 = s2.length();
		int l3 = s3.length();
		if (l1 + l2 != l3)
			return false;
        boolean[][] dp = new boolean[l1+1][l2+1];
        dp[0][0] = true;
        for (int i = 1; i<=l1; i++){
        	dp[i][0] = dp[i-1][0] && s1.charAt(i-1)==s3.charAt(i-1);
        }
        for (int j = 1; j<=l2; j++){
        	dp[0][j] = dp[0][j-1] && s2.charAt(j-1)==s3.charAt(j-1);
        }
        for (int i = 1; i<=l1; i++){
        	for (int j = 1; j<=l2; j++){
        		dp[i][j] = (dp[i-1][j]&& s1.charAt(i-1) == s3.charAt(i+j-1))
        				|| (dp[i][j-1] && s2.charAt(j-1) == s3.charAt(i+j-1));
        	}
        }
        return dp[l1][l2];
    }
	
	public int numDistinct(String s, String t) {
		if (s==null||t==null ||s.length()==0||t.length()==0)
			return 0;
        int l1 = s.length();
        int l2 = t.length();
        int[][] dp = new int[l2][l1];
        dp[0][0] = (s.charAt(0)==t.charAt(0))?1: 0;
        for (int i = 0; i<l2; i++) {
        	for (int j = i; j<l1; j++) {
        		if (i==0 && j==0) continue;
        		if (i==0) {
        			dp[0][j] = (t.charAt(i)==s.charAt(j))? dp[0][j-1]+1: dp[0][j-1];
        			continue;
        		}
        		if (t.charAt(i)==s.charAt(j)){
        			dp[i][j] = dp[i-1][j-1]+ dp[i][j-1];
        			continue;
        		}
        		dp[i][j] = dp[i][j-1];
        	}
        }
        return dp[l2-1][l1-1];
    }
	
	public boolean isScramble(String s1, String s2) {
		int length = s1.length();
		boolean[][][] DP = new boolean[length][length][length+1];
		for (int n = 1; n<=length; n++) {
			for (int i = 0; i<=length-n; i++) {
				for (int j = 0; j<=length-n; j++) {
					if (n==1 && s1.charAt(i)==s2.charAt(j)){
						DP[i][j][n] = true;
						continue;
					}
					for (int k = 1; k<n; k++){
						if ((DP[i][j][k] && DP[i+k][j+k][n-k])||
								(DP[i][j+n-k][k]&&DP[i+k][j][n-k])){
							DP[i][j][n] = true;
							break;
						}
							
					}
					
				}
			}
		}
		return DP[0][0][length];
    }
	
	private int calculateCoint(int[] nums, int balloon, int left, int right) {

		if (left < 0)
			left = 1;
		else
			left = nums[left];
		if (right >= nums.length)
			right = 1;
		else
			right = nums[right];
		return nums[balloon] * left * right;
	}
	
	public boolean wordBreak(String s, Set<String> wordDict) {
		if (s==null)
			return false;
		if (wordDict.contains(s))
			return true;
        int n = s.length();
        boolean[][] dp = new boolean[n+1][n+1];
        for (int offset = 0; offset <=n; offset ++){
        	for (int i = 0; i <= n-offset; i++) {
        		int j = i+offset;
        		if (wordDict.contains(s.substring(i, j)))
        			dp[i][j] = true;
        		else {
        			for (int x = i+1; x<j; x++) {
        				if (dp[i][x] && dp[x][j]) {
        					dp[i][j] = true;
        					break;
        				}
        			}	
        		}
        	}
        }
        return dp[0][n];
    }
	
	public int candy (int[] ratings) {
		int[] ret = new int[ratings.length];
		ret[0] = 1;
		for (int i = 1; i<ratings.length; i++) {
			if (ratings[i]>ratings[i-1])
				ret[i] = ret[i-1]+1;
			else
				ret[i] = 1;
		}
		System.out.println(Arrays.toString(ret));
		for (int i = ratings.length-2; i>=0; i--){
			if (ratings[i]>ratings[i+1] && ret[i] <= ret[i+1])
				ret[i] = ret[i+1]+1;
		}
		int sum = 0;
		for (int candies: ret) {
			sum += candies;
		}
		System.out.println(Arrays.toString(ret));
		return sum;
	}
	
	public int candyDP (int[] ratings) {
		int[] ret = new int[ratings.length];
		int sum = 0;
		for (int i = 0; i < ret.length; i++) {
			if (i!=0 && ratings[i]>ratings[i-1])
				ret[i] = ret[i-1] + 1;
			if (i!=ratings.length-1 && ratings[i]>ratings[i+1])
				ret[i] = Math.max(ret[i], 2);
			ret[i] = Math.max(ret[i], 1);
			sum += ret[i];
		}
		return sum;
	}
	
	public static void printTable (int[][] table) {
    	for (int i = 0; i<table.length; i++) {
    		for (int j = 0; j<table[0].length; j++){
    			System.out.print(table[i][j] +" ");
    		}
    		System.out.println("");
    	}
    }
	
	
	public int[] largestDivisibleSubset(int[] nums) {
		if (nums.length == 0)
			return new int[0];
        Arrays.sort(nums);
        int[] dp = new int[nums.length];
        int[] des = new int[nums.length];
        dp[0] = 1;
        int maxInd = 0;
        for (int i = 1; i < nums.length; i++) {
        	des[i] = i;
        	for (int j = 0; j < i; j++) {
        		if (nums[i] % nums[j] == 0) {
        			if (dp[j] + 1 > dp[i]) {
        				dp[i] = dp[j] + 1;
        				des[i] = j;
        			}
        		}
        	}
        	if (dp[i] > dp[maxInd]) {
        		maxInd = i;
        	}
        }
        int[] ret = new int[dp[maxInd]];
        int i = 0;
        while (i < ret.length) {
        	ret[i++] = nums[maxInd];
        	maxInd = des[maxInd];
        }
        return ret;
    }
	
	

	
	
	public static void printTable (char[][] table) {
    }
	
	
	
	public boolean canWinNim(int n) {
		return !(n%4==0);
    }
	
	
	public int minCostII(int[][] costs) {
		int n = costs.length;
		if (n == 0)
			return 0;
		int k = costs[0].length;
		int[][] dp = new int[n][k];
		dp[0] = costs[0];
		for (int i = 1; i < n; i++) {
			//we want to find the max and second max of the dp[i - 1]
			int min = dp[i - 1][0];
			int secondmin = Integer.MAX_VALUE;
			int minInd = 0;
			for (int j = 1; j < k; j++) {
				if (dp[i - 1][j] <= min) {
					minInd = j;
					secondmin = min;
					min = dp[i - 1][j];
				}
				else if (dp[i - 1][j] < secondmin) {
					secondmin = dp[i - 1][j];
				}
			}
			for (int j = 0; j < k; j++) {
				dp[i][j] = costs[i][j] + min;
			}
			dp[i][minInd] = secondmin + costs[i][minInd];
		}
		int minRet = Integer.MAX_VALUE;
		for (int cost: dp[n - 1]) {
			minRet = Math.min(minRet, cost);
		}
		return minRet;
    }
	
	public static void main (String[] args) {
		Solutions s = new Solutions();
		int[] test = new int[] {1, 2, 3};
    	System.out.println(Arrays.toString(s.largestDivisibleSubset(test)));
	}
	
	

}
