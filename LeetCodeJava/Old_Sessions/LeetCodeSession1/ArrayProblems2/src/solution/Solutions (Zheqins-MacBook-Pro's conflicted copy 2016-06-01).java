package solution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;

public class Solutions {
	public boolean isPalindrome(String s) {
        if (s==null || s.length() ==0)
        	return true;
        int i = 0;
        int j = s.length()-1;
        while (i<j) {
        	while (i<s.length() && !(Character.isAlphabetic(s.charAt(i))|| isNumerical(s.charAt(i))))
        		i++;
        	while (j>=0 && !(Character.isAlphabetic(s.charAt(j)) || isNumerical(s.charAt(j))))
        		j--;
        	if (i==s.length())
        		return true;
        	if (Character.toLowerCase(s.charAt(i))!= Character.toLowerCase(s.charAt(j)))
        		return false;
        	i++;
        	j--;
        }
        return true;
    }
	
	public int shortestDistance(String[] words, String word1, String word2) {
		HashMap<String, List<Integer>> word_dict = new HashMap<>();
		for (int i = 0; i < words.length; i++) {
			List<Integer> list = word_dict.containsKey(words[i])? word_dict.get(words[i]):
				new ArrayList<Integer>();
			list.add(i);
			word_dict.put(words[i], list);
		}
		List<Integer> list1 = word_dict.get(word1);
		List<Integer> list2 = word_dict.get(word2);
		int i = 0; 
		int j = 0;
		int minDist = Integer.MAX_VALUE;
		while (i < list1.size() && j < list2.size()) {
			int idx1 = list1.get(i);
			int idx2 = list2.get(j);
			minDist = Math.min(minDist, Math.abs(idx1 - idx2));
			if (idx1 >= idx2)
				j++;
			else
				i++;
		}
		return minDist;
    }

	
	
	
	public boolean isSelfCrossing(int[] x) {
        if (x.length < 3)
        	return true;
        for (int i = 3; i < x.length; i ++) {
        	//first check if this line is crossing the line that i-3
        	if (x[i] >= x[i - 2] && x[i - 1] <= x[i - 3])
        		return true;
        	//check if this line is crossing with the line that 4 units away
        	if (i >= 4 && x[i] + x[i - 4] >= x[i - 2] && x[i - 1] == x[i - 3])
        		return true;
        	//check if this line is crossing with the line that 5 units
        	//very important to check if this is expanding or squeezing!!
        	//x[i - 2] > x[i - 4]
        	if (i >= 5 && x[i-2] > x[i - 4] && x[i] + x[i - 4] >= x[i - 2] 
        			&& x[i - 1] < x[i - 3] && x[i - 1] >= x[i - 3] - x[i - 5])
        		return true;
        }
        return false;
    }
	
	
	public void setZeroes(int[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        boolean isRowZero = false;
        boolean isColZero = false;
        for (int i = 0; i < m; i++) {
        	if (matrix[0][i] == 0) {
        		isRowZero = true;
        		break;
        	}
        }
        for (int i = 0; i < n; i++) {
        	if (matrix[i][0] == 0) {
        		isColZero = true;
        		break;
        	}
        }
        for (int i = 1; i < n; i++) {
        	for (int j = 1; j < m; j++) {
        		if (matrix[i][j] == 0) {
        			matrix[i][0] = 0;
        			matrix[0][j] = 0;
        		}
        	}
        }
        
        printTable (matrix);
        for (int i = 1; i < n; i++) {
        	if (matrix[i][0] == 0) {
        		for (int j = 1; j < m; j++) {
        			matrix[i][j] = 0;
        		}
        	}
        }
        for (int j = 1; j < m; j++){
        	if (matrix[0][j] == 0) {
        		for (int i = 1; i < n; i++) {
        			matrix[i][j] = 0;
        		}
        	}
        }
        if (isColZero) {
        	for (int i = 0; i < n; i++) {
        		matrix[i][0] = 0;
        	}
        }
        if (isRowZero) {
        	for (int j = 0; j < m; j++) {
        		matrix[0][j] = 0;
        	}
        }
    }
	
	// 7 5 1 2    ... add 4         max 7
	// 5 1 2 4    ... add -1        max 5
	// 1 2 4 -1   ...               max 1
	public int[] maxSlidingWindow(int[] nums, int k) {
		if (nums == null || nums.length == 0 || k == 0)
            return new int[0];
		
        LinkedList<Integer> deque = new LinkedList<>();
        int[] maxWind = new int[nums.length - k + 1];
        for (int i = 0; i < nums.length; i++) {
        	int thresh = i - (k-1);
        	if (!deque.isEmpty() && deque.peekFirst() < thresh)
        		deque.removeFirst();
        	while (!deque.isEmpty() && nums[deque.peekLast()] <= nums[i])
        		deque.removeLast();
        	deque.addLast(i);
        	if (i >= k - 1)
        		maxWind[i - k + 1] = nums[deque.peekFirst()];
        }
        return maxWind;
    }
	
	public String reverseVowels(String s) {
        Set<Character> vowels = new HashSet<Character> ();
        vowels.add('a');
        vowels.add('e');
        vowels.add('i');
        vowels.add('o');
        vowels.add('u');
        vowels.add('A');
        vowels.add('E');
        vowels.add('I');
        vowels.add('O');
        vowels.add('U');
        int i = 0;
        int j = s.length() - 1;
        char[] s_arr = s.toCharArray();
        while (i < j) {
        	while (i < s_arr.length && !vowels.contains(s_arr[i]))
        		i++;
        	while (j >= 0 && !vowels.contains(s_arr[j]))
        		j--;
        	if (i < s_arr.length && j >= 0 && i < j) {
        		char temp = s_arr[i];
        		s_arr[i++] = s_arr[j];
        		s_arr[j--] = temp;
        	}
        }
        return String.valueOf(s_arr);
    }
	
	public int[][] generateMatrix(int n) {
		int[][] ret = new int[n][n];
		fillSpiralMatrix(1, 0, ret, n);
		return ret;
    }
	
	public void fillSpiralMatrix(int start, int offset, int[][] matrix, int n) {
		if (offset > n/2)
			return;
		for (int j = offset; j < n-offset; j++) {
			matrix[offset][j] = start++;
		}
		for (int i = offset + 1; i < n - offset; i++) {
			matrix[i][n - offset - 1] = start++;
		}
		for (int j = n - offset - 2; j >= offset; j--) {
			matrix[n - offset - 1][j] = start++;
		}
		for (int i = n - offset - 2; i > offset; i--) {
			matrix[i][offset] = start++;
		}
		fillSpiralMatrix(start, offset + 1, matrix, n);
	}
	
	public void wiggleSort(int[] nums) {
		partition(nums, nums.length/2, 0, nums.length-1);
		int median = nums[nums.length/2];;
    	int left = 0, i = 0, right = nums.length-1;
    	while (i<=right) {
    		//if the 
    		if (nums[newIndex(i, nums.length)] > median) {
    			swap(nums, newIndex(left++, nums.length), newIndex(i++, nums.length));
    		}
    		else if (nums[newIndex(i, nums.length)] < median) {
    			swap(nums, newIndex(right--, nums.length), newIndex(i, nums.length));
    		}
    		else {
    			i++;
    		}
    	}
    }
	
	class KVPair implements Comparable<KVPair> {
		int key;
		int pair;
		KVPair(int key, int pair) {
			this.key = key;
			this.pair = pair;
		}
		
		@Override
		public int compareTo(KVPair o) {
			return key - o.key;
		}
		
	}
	
	public List<Integer> topKFrequent(int[] nums, int k) {
		Map<Integer, Integer> freqCount = new HashMap<>();
		PriorityQueue<KVPair> freqs = new PriorityQueue<KVPair> ();
		for (int num : nums) {
			int freq = 1;
			if (freqCount.containsKey(num))
				freq = freqCount.get(num) + 1;
			freqCount.put(num, freq);
		}
		for (int num : freqCount.keySet()) {
			
			int freq = freqCount.get(num);
			System.out.println("Number: " + num + "  Frequency: " + freq);
			if (freqs.size() < k) {
				freqs.offer(new KVPair(freq, num));
			}
			else if (freqs.peek().key < freq) {
				freqs.poll();
				freqs.add(new KVPair(freq, num));
			}
		}
		List<Integer> result = new ArrayList<> ();
		while (!freqs.isEmpty()) {
			result.add(0, freqs.poll().pair);
		}
		return result;
    }
	
	//a function that can do the following mapping
	// 0 1 2 3 4 5 6 7
	// 1 3 5 7 0 2 4 6
	private int newIndex(int idx, int n) {
		return (1+2*idx) %(n|1);
	}
	
	public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> map1 = new HashSet<> ();
        ArrayList<Integer> shared = new ArrayList<>();
        for (int num : nums1) {
        	map1.add(num);
        }
        for (int num: nums2) {
        	if (map1.contains(num)) {
        		shared.add(num);
        		map1.remove(num);
        	}
        }
        int[] ret = new int[shared.size()];
        for (int i = 0; i< shared.size(); i++) {
        	ret[i] = shared.get(i);
        }
        return ret;
    }
	
	public int maximalRectangle(char[][] matrix) {
		if (matrix==null || matrix.length ==0 || matrix[0].length == 0)
            return 0;
		int n = matrix.length;
		int m = matrix[0].length;
        int[][] histos = new int[n][m];
        for (int i = 0; i< n; i++) {
        	for (int j = 0; j < m; j++) {
        		if (matrix[i][j] == '0')
        			continue;
        		else if (i == 0)
        			histos[i][j] = 1;
        		else
        			histos[i][j] = histos[i-1][j] + 1;
        	}
        }
        int maxRect = 0;
        for (int i = 0; i < n; i++) {
        	int maxArea = 0;
        	Stack<Integer> stack = new Stack<>();
        	int j = 0;
        	//run the max rectangle algorithm
        	while (!stack.isEmpty() || j < m) {
        		if (j < m && (stack.isEmpty() || histos[i][stack.peek()] <= histos[i][j])) {
        			stack.push(j++);
        			continue;
        		}
        		int bar = histos[i][stack.pop()];
        		int left = stack.isEmpty()? 0: stack.peek() + 1;
        		maxArea = Math.max(maxArea, (j - left) * bar);
        	}
        	maxRect = Math.max(maxArea, maxRect);
        }
        return maxRect;
    }
	
	public int hIndex(int[] citations) {
		int len = citations.length;
        int low = 0; 
        int high = len - 1;
        while (high > low) {
        	int mid = low + (high - low) / 2;
        	if (citations[mid] == len - mid)
        		return citations[mid];
        	if (citations[mid] > len - mid)
        		high = mid;
        	else
        		low = mid + 1;
        }
        return len - high;
    }
	
	public String countAndSay(int n) {
        String initial = "1";
        for (int i = 1; i < n; i++) {
        	initial = getNextCount(initial);
        }
        return initial;
    }
	
	public String getNextCount(String s) {
		StringBuilder sb = new StringBuilder();
		int count = 1;
		for (int i = 0; i < s.length(); i++){
			if (i == s.length() - 1 || s.charAt(i) != s.charAt(i + 1)) {
				sb.append(count);
				sb.append(s.charAt(i));
				count = 1;
				continue;
			}
			count++;
		}
		return sb.toString();
	}
	
	public String removeDuplicateLetters(String s) {
		int[] charCount = new int[26];
		for (int i = 0; i < s.length(); i++) {
			charCount[s.charAt(i) - 'a']++;
		}
		StringBuilder sb = new StringBuilder();
		Stack<Character> stack = new Stack<> ();
		boolean[] isInStack = new boolean[26];
		for (int i = 0; i < s.length(); i++) {
			charCount[s.charAt(i) - 'a']--;
			if (isInStack[s.charAt(i) - 'a']) {
				continue;
			}
			while (!stack.isEmpty() 
					&& stack.peek() > s.charAt(i) && charCount[stack.peek() - 'a'] > 0) {
				isInStack[stack.pop() - 'a'] = false;
			}
			isInStack[s.charAt(i) - 'a'] = true;
			stack.push(s.charAt(i));
		}
		while (!stack.isEmpty()) {
			sb.insert(0, stack.pop());
		}
		return sb.toString();
    }
	
	
//	public int hIndex(int[] citations) {
//		if (citations == null || citations.length == 0)
//			return 0;
//		if (citations.length == 1)
//			return Math.min(citations[0], 1);
//		// ascending order
//		Arrays.sort(citations);
//		//the max of the h index could be min(citation.length, citation[0])
//		//the min of the h index could be min(citation.length, citation[citation.length - 1])
//		for (int i = citations.length -1; i > 0 ; i --) {
//			//the index would be how many elements is greater than citations[i]
//			int hInd = citations.length - i;
//			if (hInd <= citations[i] && hInd >= citations[i - 1])
//				return citations.length - i;
//		}
//		return citations[0];
//	}
	
//	public int hIndex2 (int[] citations) {
//		HashMap<Integer, Integer> map = new HashMap<> ();
//		
//	}
	
	public List<List<String>> partition(String s) {
        List<List<String>> ret = new ArrayList<List<String>>();
        if (s==null ||s.length()==0)
        	return ret;
        boolean[][] isPal = new boolean[s.length()+1][s.length()+1];
        for (int i = 0; i<s.length(); i++) {
        	isPal[i][i+1] = true;
        	int p1 = i-1;
        	int p2 = i+1;
        	while (p1>=0 && p2<s.length()) {
        		if (s.charAt(p1) == s.charAt(p2)) {
        			isPal[p1--][p2+1] = true;
        			p2++;
        		}
        		else 
        			break;
        	}
        	p1 = i-1;
        	p2 = i;
        	while (p1>=0 && p2<s.length()) {
        		if (s.charAt(p1) == s.charAt(p2)) {
        			isPal[p1--][p2+1] = true;
        			p2++;
        		}
        		else
        			break;
        	}
        }
        List<String> list = new ArrayList<>();
        partitionHelper(isPal, 0, s, list, ret);
        return ret;
    }
	
	public void partition(int[] nums, int i, int start, int end) {
		int pivot = nums[i];
		swap(nums, i, start);
		int p1 = start;
		int p2 = end;
		int element = start+1;
		while (p1<p2) {
			if (nums[element]<=pivot) {
				nums[p1++] = nums[element++];
			}
			else {
				swap(nums, element, p2--);
			}
		}
		nums[p1] = pivot;
		if (p1==i) return;
		if (p1<i) partition(nums, i, p1+1, end);
		else partition(nums, i, start, p1-1);
	}
	
	public void swap(int[] nums, int i, int j){
		int temp = nums[i];
		nums[i] = nums[j];
		nums[j] = temp;
	}
	
//	public boolean searchMatrix(int[][] matrix, int target) {
//        //first binary search in the vertical dimension
//		if (matrix.length==0 || matrix[0].length==0)
//			return false;
//		int n = matrix.length, m = matrix[0].length;
//		int lb = 0;
//		int ub = n-1;
//		while (ub>lb+1) {
//			int mid = lb + (ub-lb)/2;
//			if (matrix[mid][0] == target)
//				return true;
//			if (matrix[mid][0] > target)
//				ub = mid;
//			else
//				lb = mid;
//		}
//		//then binary search in the horizontal dimension
//		int row = target>=matrix[ub][0]? ub: lb;
//		lb = 0; ub = m-1;
//		while (ub>lb) {
//			int mid = lb + (ub - lb)/2;
//			if (matrix[row][mid] == target)
//				return true;
//			if (matrix[row][mid]>target)
//				ub = mid;
//			else
//				lb = mid+1;
//		}
//		return matrix[row][ub] ==target;
//    }
	
	public List<Integer> countSmaller(int[] nums) {
        int[] aux = Arrays.copyOf(nums, nums.length);
        Arrays.sort(nums);
        HashMap<Integer, Integer> argSort = new HashMap<Integer, Integer>();
        for (int idx = 0; idx < nums.length; idx++) {
        	argSort.put(nums[idx], idx);
        }
        Integer[] ret = new Integer[nums.length];
        // reverse order of the sum
        int[] sums = new int[aux.length + 1];
        for (int i = aux.length - 1; i >= 0; i--) {
        	int idx = argSort.get(aux[i]);
        	int sum = 0;
        	while (idx > 0) {
        		sum += sums[idx];
        		idx -= idx & -idx;
        	}
        	ret[i] = sum;
        	idx = argSort.get(aux[i]) + 1;
        	while (idx  < sums.length) {
        		sums[idx] += 1;
        		idx += idx& -idx;
        	}
        }
        return Arrays.asList(ret);
    }
	
	int[] sums;
	public void createTree (int[] nums) {
		sums = new int[nums.length+1];
		for (int i = 0; i< nums.length; i++) {
			add(nums[i], i);
		}
	}
	
	public void add(int num, int ind) {
		ind += 1;
		while (ind < sums.length){
			sums[ind] += num;
			ind += ind &(-ind);
		}
	}
	
	public String convert(String s, int numRows) {
        if (numRows == 1)
        	return s;
        int period = numRows * 2 - 2;
        StringBuilder sb = new StringBuilder();
        for (int offset = 0; offset <= period/2; offset++) {
        	for (int i = 0; i <= s.length() / period; i++) {
        		if (i * period + offset < s.length()) {
        			sb.append(s.charAt(i*period + offset));
        		}
        		if ((i + 1) * period - offset< s.length() && offset!=period - offset && offset != 0) {
        			sb.append(s.charAt((i + 1) * period - offset));
        		}
        	}
        }
        return sb.toString();
    }
	
	public int countRangeSum(int[] nums, int lower, int upper) {
		if (nums.length == 0)
			return 0;
        long[] prefixSum = new long[nums.length + 1];
        long[] sortedSum = new long[nums.length + 1];
        for (int i = 1; i <= nums.length; i ++) {
        	prefixSum[i] = prefixSum[i - 1] + nums[i - 1];
        	sortedSum[i] = prefixSum[i];
        }
        Arrays.sort(sortedSum);
        int[] countings = new int[sortedSum.length + 1];
        int totalSum = 0;
        for (int i = prefixSum.length - 1; i >= 0; i --) {
        	//prefixSum[j] - prefixSum[i] = sum(nums[i] -----nums[j - 1])
        	//so we are looking for idx j > i and 
        	// prefixSum[i] + lower <= prefixSum[j] <= prefixSum[i] + upper
        	long ub = prefixSum[i] + upper;
        	long lb = prefixSum[i] + lower;
        	//find the highest index in sortedSum that is less than equal to ub
        	int ub_idx = binarySearch(sortedSum, ub);
        	ub_idx = sortedSum[ub_idx] > ub? ub_idx : ub_idx + 1;
        	//find the lowest index in sortedSum that is greater than or equal to lb
        	int lb_idx = binarySearch(sortedSum, lb);
        	lb_idx = sortedSum[lb_idx] < lb? lb_idx + 1: lb_idx;
        	totalSum += sum(countings, ub_idx) - sum (countings, lb_idx);
        	update(countings, binarySearch(sortedSum, prefixSum[i]) + 1);
        }
        return totalSum;
    }
	
	public int sum (int[] count, int idx) {
		int sum = 0;
		while (idx > 0) {
			sum += count[idx];
			idx -= (idx & (-idx));
		}
		return sum;
	}
	
	public void update (int[] count, int idx) {
		while (idx < count.length) {
			count[idx] += 1;
			idx += (idx & (-idx));
		}
	}
	//this is to find the lowest index that is greater than equal to the target
	public int binarySearch (long[] nums, long target) {
		int lowBound = 0;
		int highBound = nums.length - 1;
		while (highBound > lowBound) {
			int mid = lowBound + (highBound - lowBound) / 2;
			if (nums[mid] >= target) {
				highBound = mid;
			}
			else {
				lowBound = mid + 1;
			}
		}
		return lowBound;
	}

	
	public int removeElement(int[] nums, int val) {
        int offset = 0;
        for (int i = 0; i < nums.length; i++) {
        	if (nums[i] == val) {
        		offset++;
        	}
        	else {
        		nums[i - offset] = nums[i];
        	}
        }
        return nums.length - offset;
    }
	
	public List<int[]> getSkyline(int[][] buildings) {
        PriorityQueue<Building> stPQ = new PriorityQueue<>(new Comparator<Building> (){
			@Override
			public int compare(Building o1, Building o2) {
				if (o1.st != o2.st)
					return o1.st - o2.st;
				else
					return o2.height - o1.height;
			}
        });
        PriorityQueue<Building> endPQ = new PriorityQueue<>(new Comparator<Building> (){
			@Override
			public int compare(Building o1, Building o2) {
				if (o1.end != o2.end)
					return o1.end - o2.end;
				else
					return o1.height - o2.height;
			}
        });
        TreeSet<Building> heightPQ = new TreeSet<>(new Comparator<Building> (){
			@Override
			public int compare(Building o1, Building o2) {
				if(o1.height != o2.height)
					return o1.height - o2.height;
				else if (o1.st != o1.st)
					return o1.st - o2.st;
				else {
					return o1.end - o2.end;
				}
			}
        });
        for (int[] buildingArr : buildings) {
        	Building building = new Building(buildingArr);
        	stPQ.add(building);
        	endPQ.add(building);
        }
        List<int[]> ret = new ArrayList<> ();
        while (!stPQ.isEmpty()) {
        	if (stPQ.peek().st < endPQ.peek().end) {
        		Building next = stPQ.poll();
        		if (heightPQ.isEmpty() || next.height > heightPQ.last().height) {
        			ret.add(new int[]{next.st, next.height});
        		}
        		//add this into the treeset
        		heightPQ.add(next);
        	}
        	else if (stPQ.peek().st > endPQ.peek().end) {
        		Building next = endPQ.poll();
        		if (next == heightPQ.last()) {
        			heightPQ.remove(next);
        			if (heightPQ.isEmpty() || heightPQ.last().height!= next.height) {
        				int h = heightPQ.isEmpty()? 0 : heightPQ.last().height;
        				ret.add(new int[]{next.end, h});
        			}
        		}
        		else {
        			heightPQ.remove(next);
        		}
        	}
        	else {
        		Building addNext = stPQ.poll();
        		Building removeBefore = endPQ.poll();
        		if (addNext.height != removeBefore.height && removeBefore == heightPQ.last()) {
        			heightPQ.remove(removeBefore);
        			if (heightPQ.isEmpty() || heightPQ.last().height!= removeBefore.height) {
        				int h = heightPQ.isEmpty()? addNext.height: Math.max(heightPQ.last().height, addNext.height);
        				ret.add(new int[]{removeBefore.end, h});
        			}
        		}
        		else
        			heightPQ.remove(removeBefore);
        		heightPQ.add(addNext);
        	}
        }
        while (!endPQ.isEmpty()) {
        	Building next = endPQ.poll();
    		if (next == heightPQ.last()) {
    			heightPQ.remove(next);
    			if (heightPQ.isEmpty() || heightPQ.last().height!= next.height) {
    				int h = heightPQ.isEmpty()? 0 : heightPQ.last().height;
    				ret.add(new int[]{next.end, h});
    			}
    		}
    		else {
    			heightPQ.remove(next);
    		}
        }
        return ret;
    }
	
	public class Building {
		int st;
		int end;
		int height;
		
		Building(int[] descrip) {
			st = descrip[0];
			end = descrip[1];
			height = descrip[2];
		}
	}
	
	public void nextPermutation(int[] nums) {
        int i = nums.length - 1;
        while (i > 0 && nums[i] <= nums[i - 1]) {
        	i--;
        }
        if (i != 0) {
        	int pivot = i - 1;
        	while (i < nums.length && nums[i] > nums[pivot]) {
        		i ++;
        	}
        	int swap = i -1;
        	int temp = nums[swap];
        	nums[swap] = nums[pivot];
        	nums[pivot] = temp;
        	i = pivot + 1;
        	int j = nums.length - 1;
        	while (i < j) {
        		temp = nums[i];
        		nums[i] = nums[j];
        		nums[j] = temp;
        		i++;
        		j--;
        	}
        }
        else {
        	 i = 0;
        	 int j = nums.length - 1;
        	 while (i < j) {
         		int temp = nums[i];
         		nums[i] = nums[j];
         		nums[j] = temp;
         		i++;
         		j--;
        	 }
        }
    }


	
	public int sum(int ind) {
		ind += 1;
		int sum = 0;
		while (ind > 0){
			sum += sums[ind];
			ind -= ind&(-ind);
		}
		return sum;
	}

	
	public List<Integer> spiralOrder(int[][] matrix) {
		if (matrix.length==0 || matrix[0].length==0)
			return new ArrayList<Integer>();
        return spiralOrder(matrix, 0);
    }
	
	public List<Integer> spiralOrder(int[][] matrix, int offset) {
		int rMin = offset;
		int rMax = matrix.length-1-offset;
		int cMin = offset;
		int cMax = matrix[0].length-1-offset;
		List<Integer> list = new ArrayList<>();
		if (rMin>rMax || cMin>cMax)
			return list;
		for (int i = cMin; i<=cMax; i++) {
			list.add(matrix[rMin][i]);
		}
		if (rMin==rMax)
			return list;
		for (int i = rMin+1; i<=rMax; i++) {
			list.add(matrix[i][cMax]);
		}
		if (cMin==cMax)
			return list;
		for (int i = cMax-1; i>=cMin; i--){
			list.add(matrix[rMax][i]);
		}
		for (int i = rMax-1; i>=rMin+1; i--){
			list.add(matrix[i][cMin]);
		}
		list.addAll(spiralOrder(matrix, offset+1));
		return list;
	}
	
	
	private void partitionHelper(boolean[][] isPal, int start, String s, List<String> list,
			List<List<String>> ret) {
		if (start>=s.length()) {
			ArrayList<String> listCopy = new ArrayList<>();
			listCopy.addAll(list);
			ret.add(listCopy);
			return;
		}
		for (int i = start+1; i<=s.length(); i++) {
			if (isPal[start][i]) {
				list.add(s.substring(start, i));
				partitionHelper(isPal, i, s, list, ret);
				list.remove(list.size()-1);
			}	
		}
	}
	
	public static void printTable (int[][] table) {
    	for (int i = 0; i<table.length; i++) {
    		for (int j = 0; j<table[0].length; j++){
    			System.out.print(table[i][j] +" ");
    		}
    		System.out.println("");
    	}
    }
	
	public boolean isNumerical (char c) {
		return c-'0' <10 &&c-'0'>=0;
	}
	
	public String minWindow(String s, String t) {
		if (s == null || t == null || s.length() == 0 || t.length() == 0)
			return "";
		HashMap<Character, Integer> set = new HashMap<Character, Integer>();
		for (int i = 0; i < t.length(); i++) {
			int freq = set.containsKey(t.charAt(i))? set.get(t.charAt(i)) + 1: 1;
			set.put(t.charAt(i), freq);
		}
		int i = 0;
		HashMap<Character, Integer> map = new HashMap<> ();
		while (i < s.length() && !set.containsKey(s.charAt(i))) {
			i++;
		}
		if (i == s.length())
			return "";
		map.put(s.charAt(i), 1);
		int j = i + 1;
		int found = 1;
		//want to find the subsequence that contains all
		while (found != t.length() && j < s.length()) {
			char c = s.charAt(j);
			if (set.containsKey(c)) {
				int freq = map.containsKey(s.charAt(j))? map.get(s.charAt(j)) + 1: 1;
				map.put(s.charAt(j), freq);
				if (set.get(c) >= map.get(c))
					found++;
				i = movePreToFront(map, s, i, j, set);
			}
			j++;
		}
		if (j == s.length() && found != t.length())
			return  "";
		String minS = s.substring(i, j);
		while (j < s.length()) {
			char surf = s.charAt(j);
			if (map.containsKey(surf)) {
				map.put(surf, map.get(surf) + 1);
				i = movePreToFront(map, s, i, j, set);
				if (j - i + 1 < minS.length())
					minS = s.substring(i, j + 1);
			}
			j++;
		}
		return minS;
    }
	
	public int[] intersect(int[] nums1, int[] nums2) {
		HashMap<Integer, Integer> map1 = new HashMap<>();
		HashMap<Integer, Integer> map2 = new HashMap<>();
		for (int num : nums1) {
			int freq = map1.containsKey(num)? map1.get(num) + 1: 1;
			map1.put(num, freq);
		}
		for (int num : nums2) {
			int freq = map2.containsKey(num)? map2.get(num) + 1: 1;
			map2.put(num, freq);
		}
		ArrayList<Integer> ret_list = new ArrayList<> (); 
		for (int key: map1.keySet()) {
			if (map2.containsKey(key)) {
				for (int i = 0; i< Math.min(map1.get(key), map2.get(key)); i++) {
					ret_list.add(key);
				}
			}
		}
		int[] ret = new int[ret_list.size()];
		int i = 0;
		for (int num: ret_list) {
			ret[i++] = num;
		}
		return ret;
    }
	
	public boolean searchMatrix(int[][] matrix, int target) {
		if (matrix.length == 0 || matrix[0].length == 0)
			return false;
		int n = matrix.length;
		int m = matrix[0].length;
        //search the first row
		int rowLB = 0;
		int rowUB = n - 1;
		int colLB = 0;
		int colUB = m - 1;
		return searchMatrix(matrix, rowLB, rowUB, colLB, colUB, target);
	
    }
	
	public boolean searchMatrix(int[][] matrix, int rL, int rU, int cL, int cU, int target) {
		if (rL > rU || cL > cU)
			return false;
		if (matrix[rL][cL] > target || matrix[rU][cU] < target)
			return false;
		int midR = rL + (rU - rL) / 2;
		int midC = cL + (cU - cL) / 2;
		int pivot = matrix[midR][midC];
		if (pivot == target)
			return true;
		if (pivot < target) {
			//despose all the element in the upper left corner
			return searchMatrix(matrix, midR + 1, rU, midC + 1, cU, target)
					|| searchMatrix(matrix, midR + 1, rU, cL, cU, target)
					|| searchMatrix(matrix, rL, rU, midC + 1, cU, target);
		}
		else {
			//depose all the element in the lower right corner
			return searchMatrix(matrix, rL, midR - 1, cL, midC - 1, target)
					|| searchMatrix(matrix, rL, midR - 1, cL, cU, target)
					|| searchMatrix(matrix, rL, rU, cL, midC - 1, target);
		}
	}
	
	
	public int movePreToFront (HashMap<Character, Integer> map, String s,
			int i, int j, HashMap<Character, Integer> set) {
		while (i < j) {
			char pre = s.charAt(i);
			if (!map.containsKey(pre))
				i++;
			else if (map.get(pre) > set.get(pre)) {
				map.put(pre, map.get(pre) - 1);
				i++;
			}
			else
				break;
		}
		return i;
	}
	
	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
		int totalLength = nums1.length + nums2.length;
		if (totalLength % 2 == 1) {
			return findKSortedArrays(nums1, nums2, 0, nums1.length-1,0 ,  nums2.length-1, totalLength/2);
		}
		else {
			return 0.5*(findKSortedArrays(nums1, nums2, 0, nums1.length-1,0 ,  nums2.length-1, totalLength/2)
					+findKSortedArrays(nums1, nums2, 0, nums1.length-1,0 ,  nums2.length-1, totalLength/2-1));
		}
    }
	
	
	//Really hard one!!
	public int findKSortedArrays(int[] nums1, int[] nums2, int start1, int end1,
			int start2, int end2, int k) {
		if (end1 < start1)
			return nums2[start2 + k];
		if (end2 < start2)
			return nums1[start1 + k];
		if (k==0)
			return Math.min(nums1[start1], nums2[start2]);

		int mid1 = start1 + (end1- start1) / 2;
		int mid2 =  start2 + (end2 - start2) / 2;
		int n = mid1 - start1 + mid2 - start2 + 1;
		//first array is the bigger
		if (nums1[mid1] >= nums2[mid2]) {
			//if n is bigger than k, which means we are looking for the smaller things
			//in this case, we can dispose the bigger half of the first array
			//the elements that has been disposed is mid1 ---- end1, which is end1-mid1+1
			//but we do not need to shrink the k by the size of this number
			if (k < n)
				return findKSortedArrays(nums1, nums2, start1, mid1-1, start2, end2, k);
			
//			if (k == n)
//				return findKSortedArrays(nums1, nums2, start1, mid1, start2, end2, k);
			//if n is smaller than or equal to k, which means we are looking for the bigger
			//elements
			//we need to shrink the size of k to recurse
			if (k >= n)
				return findKSortedArrays(nums1, nums2, start1, end1, mid2 + 1, end2, k - (mid2 - start2 + 1));
		}
		else {
			//same approach
			if (k < n)
				return findKSortedArrays(nums1, nums2, start1, end1, start2, mid2 - 1, k);
//			if (k == n)
//				return findKSortedArrays(nums1, nums2, start1, end1, start2, mid2, k);
			else
				return findKSortedArrays(nums1, nums2, mid1+1, end1, start2, end2, k - (mid1 - start1 + 1));
		}
		return -1;
	}
	
	public boolean search(int[] nums, int target) {
        return rotatedSearch(nums, target, 0, nums.length - 1);
    }
	
	public boolean binarySearch(int[] nums, int target, int lb, int ub){
		if (lb == ub)
			return nums[lb] == target;
		int mid = lb + (ub - lb) / 2;
		if (nums[mid] < target)
			return binarySearch(nums, target, mid + 1, ub);
		return binarySearch(nums, target, lb, mid);
		
	}
	
	public boolean rotatedSearch(int[] nums, int target, int lb, int ub) {
		if (lb == ub)
			return target == nums[lb];
		if (nums[lb] < nums[ub])
			return binarySearch(nums, target, lb, ub);
		int mid = lb + (ub - lb) / 2;
		// 1 2 3 4 5
		// 4 5 1 2 3
		//sorted array on the left side
		if (nums[mid] > nums[ub]) {
			if (target > nums[lb] && target < nums[mid])
				return binarySearch(nums, target, lb, mid);
			if (target < nums[ub] || target > nums[mid])
				return rotatedSearch(nums, target, mid+1, ub);
			return rotatedSearch(nums, target, lb, mid) || rotatedSearch(nums, target, mid+1, ub);
		}
		//1 3 3 3 3 3 3
		//3 3 1 3 3 3 3
		//3 3 3 3 3 1 3
		//this case, the right side is sorted
		//this case nums[mid] == nums[lb] == nums[ub]
		else if (nums[mid] < nums[lb]){
			if (target > nums[lb] || target < nums[mid])
				return rotatedSearch(nums, target, lb, mid);
			if (target > nums[mid] && target < nums[ub])
				return binarySearch(nums, target, mid+1, ub);
			return rotatedSearch(nums, target, lb, mid) || rotatedSearch(nums, target, mid+1, ub);
		}
		return rotatedSearch(nums, target, lb, mid) || rotatedSearch(nums, target, mid+1, ub);
	}

	
	public static void main(String[] args) {
		Solutions s = new Solutions();
		int[] test = new int[] { -2, 5, -1};
		
		System.out.println(s.countRangeSum(test, -2, 2));
//		for (int[] point: s.getSkyline(test)) {
//			System.out.println(point[0] + ", " + point[1]);
//		}
//		System.out.println(s.findMedianSortedArrays(sorted1, sorted2));
//		for (int i : s.countSmaller(test)) {
//			System.out.println(i);
//		}
	}
}
