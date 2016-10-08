import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Stack;

public class Solution_1_to_50 {
	
	//start1-----mid1-------end1
	//start2-----mid2-------end2
	//mid2 > mid1 : atleast mid1 + start1 + 1 + mid 2 - start2 less than mid2
	// if this number is greater than (nums1.length + nums2.length) /2
	// skip the larger side, if less than or equal, skip the smaller side, if
	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
		int total = nums1.length + nums2.length;
		if (total % 2 == 1)
			return findKthSortedArraysHelper(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, total / 2 + 1);
		else
			return (findKthSortedArraysHelper(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, total / 2 + 1)
					+ findKthSortedArraysHelper(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, total / 2)) /2.0;
    }
	
	public void nextPermutation(int[] nums) {
        int j = nums.length - 1;
        int prev = Integer.MIN_VALUE;
        while (j >= 0) {
            if (nums[j] < prev)
                break;
            prev = nums[j--];
        }
        if (j < 0) {
            swapRange(nums, 0, nums.length - 1);
            return;
        }
        int i = j + 1;
        while (i < nums.length) {
            if (nums[i++] < nums[j]) {
                break;
            }
        }
        i--;
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
        swapRange(nums, j + 1, nums.length - 1);
    }
    
    public void swapRange(int[] nums, int i, int j) {
        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
    }
    
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> rv = new ArrayList<>();
        dfs(rv, nums, 0);
        return rv;
    }
    
    public void dfs (List<List<Integer>> rv, int[] nums, int i) {
    	//i is the element that in the head
    	//efficiently saying it is 0 -- i + permuations(i + 1)
    	if (i == nums.length) {
    		ArrayList<Integer> list = new ArrayList<>();
    		for (int num: nums) {
    			list.add(num);
    		}
    		rv.add(list);
    	}
        for (int j = i; j < nums.length; ++j) {
        	swap(nums, i, j);
        	dfs(rv, nums, i + 1);
        	swap(nums, i, j);
        }
        return;
    }
    
    public int firstMissingPositive(int[] nums) {
    	int i = 0;
        while (i < nums.length) {
            if (nums[i] != i + 1 && nums[i] <= nums.length && nums[i] > 0 && nums[i] != nums[nums[i] - 1]) {
            	swap(nums, nums[i] - 1, i);
            }
            else {
            	i++;
            }
        }

        for (i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }
    
    public void swap (int[] nums, int i, int j) 
    {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
	
//	public boolean isMatch(String s, String p) {
//        int len_s = s.length();
//        int len_p = p.length();
//        boolean[][] dp = new boolean[len_s + 1][len_p + 1];
//        dp[0][0] = true;
//        for (int j = 1; j <= len_p; ++j) {
//            if (p.charAt(j - 1) == '*') {
//                dp[0][j] = dp[0][j - 1];
//            }
//        }
//        for (int i = 1; i <= len_s; i++) {
//            for (int j = 1; j <= len_p; j++) {
//                //s: xxxx s[i - 1]
//                //p: xxxx p[j - 1]
//                if (s.charAt(i - 1) == p.charAt(j - 1)) {
//                    dp[i][j] = dp[i - 1][j - 1];
//                    continue;
//                }
//                if (p.charAt(j - 1) == '.') {
//                    dp[i][j] = dp[i - 1][j - 1];
//                    continue;
//                }
//                //s: xxxx            s[i - 1]
//                //p: xxxx p[j - 2]     *
//                if (p.charAt(j - 1) == '*') {
//                    if (p.charAt(j - 2) == s.charAt(i - 1) || p.charAt(j - 2) == '.'){
//                        dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
//                    }
//                    else {
//                        dp[i][j] = dp[i][j - 2];
//                    }
//                }
//            }
//        }
//        return dp[len_s][len_p];
//    }
	
	//abcabcd
	//[0, 
	
	public boolean isValid(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        map.put('(', 1);
        map.put('[', 2);
        map.put('{', 3);
        map.put(')', -1);
        map.put(']', -2);
        map.put('}', -3);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
        	char c = s.charAt(i);
        	if (map.containsKey(c)) {
        		int rep = map.get(c);
        		if (rep > 0) {
        			stack.push(rep);
        		}
        		else if (stack.isEmpty()) {
        			return false;
        		}
        		else {
        			int pop = stack.pop();
        			if (pop != -rep) {
        				return false;
        			}
        		}
        	}
        }
        return stack.isEmpty();
    }
	
	public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) {
            return "";
        }
        int pre = 0;
        while (true) {
            char c = '\0';
            for (String str: strs) {
                if (pre == str.length()) {
                    return strs[0].substring(0, pre);
                }
                if (c == '\0') {
                    c = str.charAt(pre);
                }
                else if (c != str.charAt(pre)) {
                    return strs[0].substring(0, pre);
                }
            }
            pre++;
        }
    }
	
	public int findKthSortedArraysHelper(int[] nums1, int s1, int e1,
			int[] nums2, int s2, int e2, int k) {
		System.out.println("s1: " + s1 + " e1: " + e1 + " s2: " + s2 + " e2: " + e2 + " k: " + k);
		if (s1 > e1)
			return nums2[s2 + k - 1];
		if (s2 > e2)
			return nums1[s1 + k - 1];
		if (k == 1)
			return Math.min(nums1[s1], nums2[s2]);
		int mid1 = s1 + (e1 - s1) / 2;
		int mid2 = s2 + (e2 - s2) / 2;
		if (nums1[mid1] > nums2[mid2]) {
			//less than nums1[mid1]
			int less = mid2 - s2 + 1 + mid1 - s1;
			if (less >=k) {
				return findKthSortedArraysHelper(nums1, s1, mid1 - 1, nums2, s2, e2, k);
			}
			else {
				return findKthSortedArraysHelper(nums1, s1,  e1, nums2, mid2 + 1, e2, k - (mid2 - s2 + 1));
			}
		}
		else if (nums1[mid1] < nums2[mid2]) {
			int less = mid1 - s1 + 1 + mid2 - s2;
			if (less >= k)
				return findKthSortedArraysHelper(nums1, s1, e1, nums2, s2, mid2 - 1, k);
			else {
				return findKthSortedArraysHelper(nums1, mid1 + 1, e1, nums2, s2, e2, k - (mid1 -s1 + 1));
			}
		}
		else {
			int less = mid1 - s1 + mid2 - s2;
			if (less == k - 1 || less == k - 2)
				return nums1[mid1];
			if (less < k - 2)
				return findKthSortedArraysHelper(nums1, mid1 + 1, e1, nums2,  mid2 + 1, e2, k - less - 2);
			else 
				return findKthSortedArraysHelper(nums1, s1, mid1 - 1, nums2, s2, mid2 - 1, k);		
		}
	}
	
	
	
	public boolean isPalindrome(int x) {
		if (x == Integer.MIN_VALUE)
			return false;
		if (x == 0)
			return true;
        int power = 1;
        while (x /power >= 10) {
        	power *= 10;
        }
        while (power > 1) {
        	if (x / power != x % 10)
        		return false;
        	x = (x % power) / 10;
        	power /= 100;
        }
        return true; 
    }
	
	
	public int[] searchRange(int[] nums, int target) {
		//first search, we want the nums[UB] >= target
		int UB = nums.length - 1;
		int LB = 0;
		while (UB >= LB) {
			int mid = LB + (UB - LB) / 2;
			if (nums[mid] < target) {
				LB = mid + 1;
			}
			else {
				UB = mid - 1;
			}
		}
		int first = UB + 1;
		if (first < 0 || first>= nums.length || nums[first] != target)
			return new int[] {-1, -1};
		//second search, we want the nums[LB] <= target
		UB = nums.length - 1;
		LB = 0;
		while (UB >= LB) {
			int mid = LB + (UB - LB) / 2;
			if (nums[mid] <= target) {
				LB = mid + 1;
			}
			else {
				UB = mid - 1;
			}
		}
		return new int[] {first, LB - 1};
    }
	
	
	public String multiply(String num1, String num2) {
		if (num1 == null || num1.length() == 0 || num2 == null || num2.length() == 0)
			return "0";
		String base = "0";
		for (int offset = 0; offset < num2.length(); offset++) {
			String singleMulti = multiplyDigit(num1, 
					num2.charAt(num2.length() - offset - 1) - '0', offset);
			base = addition(base, singleMulti);
		}
		return base;
		
    }
	
	public String multiplyDigit(String num, int c, int offset) {
		StringBuilder sb = new StringBuilder();
		int carry = 0;
		for (int digit = num.length() - 1; digit >= 0; digit--) {
			int dn = num.charAt(digit) - '0';
			int d = (dn * c + carry) % 10;
			carry = (dn * c + carry) / 10;
			sb.insert(0, String.valueOf(d));
		}
		if (carry != 0) {
			sb.insert(0, String.valueOf(carry));
		}
		if (offset != 0) {
			for (int i = 0; i < offset; i++) {
				sb.append('0');
			}
		}
		return sb.toString();
	}
	
	public String addition (String num1, String num2) {
		int carry = 0;
		StringBuilder sb = new StringBuilder();
		int digit;
		for (digit = 0; digit < Math.min(num1.length(), num2.length()); digit++) {
			int d1 = num1.charAt(num1.length() - digit - 1) - '0';
			int d2 = num2.charAt(num2.length() - digit - 1) - '0';
			int d = (d1 + d2 + carry) % 10;
			carry = (d1 + d2 + carry) / 10;
			sb.insert(0, String.valueOf(d));
		}
		while (digit < num1.length()) {
			int d1 = num1.charAt(num1.length() - digit - 1) - '0';
			int d = (d1 + carry) % 10;
			carry = (d1 + carry) / 10;
			sb.insert(0, String.valueOf(d));
			digit++;
		}
		while (digit < num2.length()) {
			int d2 = num2.charAt(num2.length() - digit - 1) - '0';
			int d = (d2 + carry) % 10;
			carry = (d2 + carry) / 10;
			sb.insert(0, String.valueOf(d));
			digit++;
		}
		if (carry == 1)
			sb.insert(0, "1");
		return sb.toString();
	}
	
	public ListNode reverseKGroup(ListNode head, int k) {
        if (k == 1 || head == null || head.next == null) {
        	return head;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p1 = dummy;
        ListNode p2 = head;
        ListNode p3 = head;
        int count = 0;
        while (p3 != null) {
        	if (count != k) {
        		p3 = p3.next;
        		count ++;
        		continue;
        	}
        	count = 0;
        	ListNode temp = p3.next;
        	reverseKNode(p2, p3);
        	p1.next = p3;
        	p2.next = temp;
        	p1 = p2;
        	p2 = p2.next;
        	p3 = p2;
        }
        return dummy.next;
    }
	
	public int reverse(int x) {
        if (x == Integer.MIN_VALUE) {
            return 0;
        }
        boolean isNegative = (x < 0);
        x = Math.abs(x);
        int rev = 0;
        while (x != 0) {
            if (rev !=  rev * 10 / 10) {
                return 0;
            }
            rev = rev * 10 + (x % 10);
            x /= 10;
        }
        return (isNegative)? -1 * rev: rev;
    }
	
	public ListNode reverseKNode (ListNode start, ListNode end) {
		if (start == end) {
			return start;
		}
		ListNode next = start.next;
		ListNode retHead = reverseKNode(start.next, end);
		start.next = null;
		next.next = start;
		return retHead;
	}
	
	public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		List<List<Integer>> rv = new ArrayList<>();
		Arrays.sort(candidates);
		if (candidates == null || candidates.length == 0)
			return rv;
		Stack<Integer> s = new Stack<Integer>();
		for (int start = 0; start < candidates.length; start++) {
			if (start != 0 && candidates[start] == candidates[start - 1])
				continue;
			s.push(candidates[start]);
			dfs(candidates, target, start, candidates[start], s,  rv);
			s.pop();
		}
		return rv;
    }
	
	public void dfs(int[] candidates, int target, int idx, 
			int sum, Stack<Integer> s, List<List<Integer>> rv) {
		 if (sum > target) {
			 return;
		 }
		 if (sum == target) {
			 ArrayList<Integer> copy = new ArrayList<>();
			 copy.addAll(s);
			 rv.add(copy);
		 }
		 for (int i = idx + 1; i < candidates.length; i++) {
			 if (i != idx + 1 && candidates[i] == candidates[idx]) {
				 continue;
			 }
			 s.push(candidates[i]);
			 dfs(candidates, target, i, sum + candidates[i], s, rv);
			 s.pop();
		 }
	}
	
	public void rotate(int[][] matrix) {
		int n = matrix.length;
		if (n < 2) {
			return;
		}
		int temp = 0;
		for (int offset = 0; offset <= n / 2; offset++) {
			for (int i = offset; i < n - offset - 1; i++) {
				//interchange matrix[offset][i] - > matrix[i][n - offset - 1]
				//- > matrix[n - offset - 1][n - i - 1] - > matrix[n - i - 1][offset]
				temp = matrix[offset][i];
				matrix[offset][i] = matrix[n - i - 1][offset];
				matrix[n - i - 1][offset] = matrix[n - offset - 1][n - i - 1];
				matrix[n - offset - 1][n - i - 1] = matrix[i][n - offset - 1];
				matrix[i][n - offset - 1] = temp;
			}
		}
		for (int i = 0; i < n; i++) {
			System.out.println(Arrays.toString(matrix[i]));
		}
    }
	
	
	public ListNode mergeKLists(ListNode[] lists) {
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>(){

			@Override
			public int compare(ListNode o1, ListNode o2) {
				// TODO Auto-generated method stub
				return o1.val - o2.val;
			}
        	
        });
        
        for (ListNode node: lists) {
        	pq.add(node);
        }
         
        while (!pq.isEmpty()) {
        	ListNode node = pq.remove();
        	p.next = node;
        	p = p.next;
        	if (node.next != null) {
        		pq.offer(node.next);
        	}
        }
        return dummy.next;
    }
	
	
	public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> rv = new ArrayList<>();
        int k = words.length;
        if (words.length == 0) {
            return rv;
        }
        int iv = words[0].length();
        if (s.length() < iv * words.length) {
            return rv;
        }
        HashMap<String, Integer> counts = new HashMap<>();
        for (String word: words) {
            int freq = counts.containsKey(word)? counts.get(word) + 1: 1;
            counts.put(word, freq);
        }
        for (int i = 0; i < iv; i++) {
        	System.out.println(i);
            HashMap<String, Integer> map = new HashMap<>();
            int start = i;
            int j = start + iv;
            while (j <= s.length()) {
            	System.out.println("start: " + start + "  end: "+ j);
                String sub = s.substring(j - iv, j);
                if (counts.containsKey(sub)) {
                    int freq = map.containsKey(sub)? map.get(sub): 0;
                    if (freq == counts.get(sub)) {
                        String remove;
                        do {
                            remove = s.substring(start, start + iv);
                            map.put(remove, map.get(remove) - 1);
                            start += iv;
                        } while (!remove.equals(sub));
                        map.put(sub, freq);
                    }
                    else {
                        map.put(sub, freq + 1);
                        if (j - start == iv * k){
                            rv.add(start);
                            String remove = s.substring(start, start + iv);
                            map.put(remove, map.get(remove) - 1);
                            start += iv;
                        }
                    }
                    j += iv;
                }
                else {
                    start = j;
                    j += iv;
                    map.clear();
                }
            }
        }
        return rv;
    }
	
	public boolean isMatch(String s, String p) {
		if (p.length() == 0) {
			return s.length() == 0;
		}
		//preprocess, break s into array
		int i = 0;
		while (i< p.length() && p.charAt(i) != '*') {
			i++;
		}
		String head = p.substring(0, i);
		if (i > s.length() || wildcardKMP(s.substring(0, i), head, 0) != 0){
			return false;
		}
		p = p.substring(i);
		s = s.substring(i);
		int j = p.length() - 1;
		while (j >= 0 && p.charAt(j) != '*') {
			j--;
		}
		String tail = p.substring(j + 1);
		p = p.substring(0, j + 1);
		if (tail.length() > s.length()) {
			return false;
		}
		String tailS = s.substring(s.length() - tail.length());
		s = s.substring(0, s.length() - tail.length());
		if (wildcardKMP(tailS, tail, 0) != 0) {
			return false;
		}
		if (p.length() == 0 && s.length() != 0)
			return false;
		i = 0;
		int start = 0;
		while (i < p.length()) {
			if (p.charAt(i) == '*') {
				i++;
				continue;
			}
			j = i + 1;
			while (j < p.length() && p.charAt(j) != '*') {
				j++;
			}
			String piece = p.substring(i, j);
			start = wildcardKMP(s, piece, start);
			if (start == -1) {
				return false;
			}
			start += piece.length();
			i = j;
		}
		return true;
		
    }
	
	
	public int[] wildcardNext(String p) {
		int[] next = new int[p.length()];
		next[0] = -1;
		for (int i = 1; i < p.length(); ++i) {
			int j = next[i - 1];
			if (p.charAt(i - 1) == '?') {
				next[i] = j + 1;
				continue;
			}
			while (j != -1 && p.charAt(j) != '?' && p.charAt(j) != p.charAt(i - 1)) {
				j = next[j];
			}
			next[i] = j + 1;
		}
		return next;
	}
	
	public int wildcardKMP(String s, String p, int start) {
		if (p.length() == 0)
			return start;
		int[] next = wildcardNext(p);
		//System.out.println(Arrays.toString(next));
		int i = start;
		int j = 0;
		while (i < s.length() && j < p.length()) {
			if (j == -1 || s.charAt(i) == p.charAt(j) || p.charAt(j) == '?') {
				i++;
				j++;
				continue;
			}
			j = next[j];
		}
		if (j == p.length()) {
			return i - j;
		}
		return -1;
	}
	
	public int trap(int[] height) {
        if (height.length < 3) {
            return 0;
        }
        int highLeft = height[0];
        int[] wall = new int[height.length];
        for (int i = 1; i < height.length; ++i) {
            wall[i] = highLeft;
            highLeft = Math.max(highLeft, height[i]);
        }
        int highRight = height[height.length - 1];
        int conserv = 0;
        for (int i = height.length - 2; i >=0; i--) {
            if (highRight > height[i] && wall[i] > height[i]) {
                int water = Math.min(highRight, wall[i]) - height[i];
                conserv += water;
            }
            highRight = Math.max(highRight, height[i]);
        }
        return conserv;
    }
	
	public int jump(int[] nums) {
        int curEnd = 0;
        int curFastest = nums[0];
        int step = 0;
        for (int i = 0; i < nums.length; i++) {
            //trigger another jump
            if (i == curEnd) {
                curEnd = curFastest;
                step++;
                if (curEnd >= nums.length - 1) {
                    return step;
                }
            }
            curFastest = Math.max(i + nums[i], curFastest);
        }
        return step;
        
    }
	
	
	public int divide(int dividend, int divisor) {
        if (dividend == 0) {
            return 0;
        }
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        if (divisor == 1) {
        	return dividend;
        }
        if (dividend == Integer.MIN_VALUE) {
            if (divisor == -1)
                return Integer.MAX_VALUE;
        }
        if (dividend > 0 && divisor < 0){
            return -1 * divide(-dividend, divisor);
        }
        if (dividend < 0 && divisor > 0) {
            return -1 * divide(dividend, -divisor);
        }
        if (divisor < dividend) {
            return 0;
        }
        int result = 1;
        int copy = divisor;
        while (copy > dividend - copy) {
            result += result;
            copy += copy;
        }
        return result + divide(dividend - copy, divisor);
    }
	
	
	public int maxArea(int[] height) {
        if (height.length < 3)   return 0;
        int[] barrier = new int[height.length];
        int left = Integer.MIN_VALUE, water = 0;
        for (int i = 0; i < height.length; ++i) {
            barrier[i] = i == 0? Integer.MIN_VALUE: left;
            left = Math.max(height[i], left);
        }
        int right = Integer.MIN_VALUE;
        for (int i = height.length - 1; i >=0; i--) {
            if (right > height[i] && barrier[i] > height[i]) {
                water += Math.min(right, barrier[i]) - height[i];
            }
            right = Math.max(right, height[i]);
        }
        return water;
    }
	
	
	public static void main (String[] args) {
		int[] nums1 = {2, 3, 1, 1, 4};
		//System.out.println(Arrays.toString(" ".split(" ")));
		Solution_1_to_50 s = new Solution_1_to_50();
		System.out.println(s.maxArea(new int[]{1, 1}));
	}

}
