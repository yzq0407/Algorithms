import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Stack;
import java.util.Comparator;


public class Solutions {
    public int findMin(int[] nums) {
    	if (nums==null||nums.length==0)
    		return -1;
        return findMin(nums, 0, nums.length-1);
    }
    
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int offset = 0; offset < n/2; offset++) {
        	for (int i = offset; i<n-offset-1; i++){
        		int aux = matrix[offset][i];
        		matrix[offset][i] = matrix[n-i-1][offset];
        		matrix[n-i-1][offset] = matrix[n-offset-1][n-i-1];
        		matrix[n-offset-1][n-i-1] = matrix[i][n-offset-1];
        		matrix[i][n-offset-1] = aux;
        	}
        }
    }
    
    public List<Integer> findSubstring(String s, String[] words) {
    	if (words.length == 0 || s.length() == 0)
    		return new ArrayList<Integer>();
        HashMap<String, Integer> dict = new HashMap<>();
        for (String word : words) {
        	int freq = dict.containsKey(word)? dict.get(word): 0;
        	dict.put(word, freq + 1);
        }
        int l = words[0].length();
        List<Integer> ret = new ArrayList<>();
        for (int i = 0; i < l; i ++) {
        	// the ones we have already seen
        	HashMap<String, Integer> isVisited = new HashMap<> ();
        	int p1 = i, p2 = i, count = 0;
        	while (p2 + l <= s.length()) {
        		//System.out.println(s.substring(p2) + "      " + count);
        		String segment = s.substring(p2, p2 + l);
        		if (dict.containsKey(segment)) {
        			count++;
        			int freq = isVisited.containsKey(segment)? isVisited.get(segment) + 1: 1;
        			isVisited.put(segment, freq);
        			p2 += l;
        			//this case we have found everything in the dict
        			if (freq == dict.get(segment) && count == words.length) {
        				ret.add(p1);
        				isVisited.put(s.substring(p1, p1 + l), isVisited.get(s.substring(p1, p1 + l)) - 1);
        				p1 += l;
        				count--;
        			}
        			if (freq > dict.get(segment)) {
        				while (true) {
        					String prefix = s.substring(p1,  p1 + l);
        					isVisited.put(prefix, isVisited.get(prefix) - 1);
        					p1 += l;
        					count--;
        					if (prefix.equals(segment))
        						break;
        				}
        			}
        		}
        		//if there is no such element, dispose everything in the map
        		else {
        			count = 0;
        			p2 += l;
        			p1 = p2;
        			isVisited = new HashMap<String, Integer>();
        		}
        	}
        }
        return ret;
    }
    
    public void findSubstringHelper(String s, HashMap<String, Integer> dict, int idx,
    		List<Integer> ret, LinkedList<Integer> list, int l, HashMap<String, Integer> isVisited, int remains){
    	while (true) {
	    	if (idx + l > s.length())
	    		return;
	    	String segment = s.substring(idx, idx + l);
	    	if (dict.containsKey(segment)){
				int freq = isVisited.containsKey(segment)? isVisited.get(segment) + 1: 1; 
				isVisited.put(segment, freq);
				list.addFirst(idx);
	    		if (dict.get(segment) == freq) {
	    			if (remains == 1) {
	    				int start = list.removeLast();
	    				ret.add(start);
	    				isVisited.put(s.substring(start, start + l),
	    						isVisited.get(s.substring(start, start + l)) - 1);
	    			}
	    			else {
	    				remains--;
	    			}
	    		}
	    		else if (dict.get(segment) < freq){
	    			while (true) {
	    				int removeIdx = list.removeLast();
	    				String removed = s.substring(removeIdx, removeIdx + l);
	    				isVisited.put(removed, isVisited.get(removed) - 1);
	    				if (removed.equals(segment))
	    					break;
	    				remains++;
	    			}
	    		}
	    	}
	    	else {
	    		isVisited = new HashMap<>();
	    		remains += list.size();
	    		list.clear();
	    	}
	    	idx = idx + l;
    	}
    }
    
    
    
    
    
    public boolean isAnagram(String s, String t) {
    	if (s.length()!=t.length())
    		return false;
    	int[] map = new int[26];
    	for (int i = 0; i < s.length(); i++){
    		map[s.charAt(i) - 'a']++;
    	}
    	for (int i = 0; i< t.length(); i++){
    		if (--map[t.charAt(i) - 'a']<0)
    			return false;
    	}
    	return true;
    }
    
//    public List<int[]> getSkyline(int[][] buildings) {
////        PriorityQueue<int[]> leftPQ = new PriorityQueue<> (new Comparator<int[]> (){
////			@Override
////			public int compare(int[] o1, int[] o2) {
////				return o1[0] - o2[0];
////			}
////        	
////        });
//        PriorityQueue<int[]> heightPQ = new PriorityQueue<> (new Comparator<int[]> (){
//
//			@Override
//			public int compare(int[] o1, int[] o2) {
//				return o1[2] - o2[2];
//			}
//        });
//        
//        List<int[]> ret = new ArrayList<int[]> ();
//        for (int[] building:buildings) {
//        	heightPQ.add(building);
//        }
//        
//        while (!heightPQ.isEmpty()){
//        	
//        }
//    }
    
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
    	int length = Math.min(C,  G)>Math.max(A,  E)? Math.min(C,  G)-Math.max(A,  E):0;
    	int height = Math.min(D, H)>Math.max(B, F)? Math.min(D, H) - Math.max(B, F):0;
    	int overlap = length*height;   	
    	return (G-E)*(H-F) - overlap + (C-A) *(D-B);   	
    }
    
    public boolean wordPattern(String pattern, String str) {
    	if (pattern==null || str==null)
    		return false;
    	String[] str_arr = str.split(" ");
    	if (pattern.length()!= str_arr.length)
    		return false;
    	HashMap<Character, String> cToSMap = new HashMap<>();
    	HashMap<String, Character> sToCMap = new HashMap<>();
    	for (int i = 0; i<pattern.length(); i++) {
    		char c = pattern.charAt(i);
    		String s = str_arr[i];
    		if (!cToSMap.containsKey(c) && !sToCMap.containsKey(s)){
    			cToSMap.put(c, s);
    			sToCMap.put(s, c);
    			continue;
    		}
    		if (cToSMap.containsKey(c) && !cToSMap.get(c).equals(s))
    			return false;
    		if (sToCMap.containsKey(s) && !sToCMap.get(s).equals(c))
    			return false;
    	}
    	return true;
    }
    
    public int search(int[] nums, int target) {
        return rotatedSearch(nums, target, 0, nums.length-1);
    }
    
    public int rotatedSearch (int[] nums, int target, int lb, int ub) {
    	if (lb>ub)
    		return -1;
    	if (nums[lb] <= nums[ub])
    		return binarySearch(nums, target, lb, ub);
    	
    	int mid = lb + (ub- lb)/2;
    	if (nums[mid] == target)
    		return mid;
    	if (nums[mid]>=nums[lb]) {
    		if (target<nums[mid] && target>=nums[lb]) {
    			return binarySearch(nums, target, lb, mid-1);
    		}
    		else
    			return rotatedSearch(nums, target, mid+1, ub);
    	}
    	else {
    		if (target>nums[mid] && target <=nums[ub])
    			return binarySearch(nums, target, mid+1, ub);
    		else
    			return rotatedSearch(nums, target, lb, mid-1);
    	}
    }
    
    public int firstMissingPositive(int[] nums) {
    	int i = 0;
        while (i<nums.length) {
        	if (nums[i]<= nums.length && nums[i]>0 && nums[i]!= i+1 && nums[nums[i]-1] != nums[i]){
        		swap (nums, i, nums[i]-1);
        	}
        	else
        		i++;
        }
        for (i = 0; i<nums.length; i++){
        	if (nums[i]>nums.length || nums[i]<=0 || nums[i]!=i+1)
        		return i+1;
        }
        return nums.length+1;
    }
    
    public int removeDuplicates(int[] nums) {
    	if (nums==null || nums.length==0)
    		return 0;
        int i = 0;
        int count = 0;
        int offset = 0;
        while (i<nums.length) {
        	if (i!=0 && nums[i] == nums[i-1]){
        		count++;
        		if (count>1) {
        			offset++;
        			i++;
        			continue;
        		}
        	}
        	else
        		count = 0;
        	nums[i-offset] = nums[i];
        	i++;
        }
        return nums.length-offset;
    }
    
    public boolean isValidSudoku(char[][] board) {
    	@SuppressWarnings("unchecked")
		List<int[]>[] map = (ArrayList<int[]>[]) new ArrayList[10];
    	for (int i = 0; i<board.length; i++){
    		for (int j = 0; j<board[0].length; j++){
    			if (Character.isDigit(board[i][j])) {
    				int number = board[i][j] - '0';
    				if (map[number]!=null){
    					List<int[]> duplicates = map[number];
    					for (int[] pairs: duplicates) {
    						if (pairs[0]==i || pairs[1] == j)
    							return false;
    						if (pairs[0]/3 == i/3 && pairs[1]/3==j/3)
    							return false;
    					}
    					duplicates.add(new int[] {i, j});
    				}
    				else {
        				map[number] = new ArrayList<int[]>();
        				map[number].add(new int[] {i, j});
        			}
    			}
    			
    		}
    	}
    	return true;
    }

    
    public int binarySearch (int[] nums, int target, int lb, int ub) {
    	if (lb>ub)
    		return -1;
    	
    	if (lb==ub)
    		return (nums[lb]==target)?lb:-1;
    	int mid = lb + (ub- lb)/2;
    	if (nums[mid] == target)
    		return mid;
    	if (nums[mid]> target)
    		return binarySearch(nums, target, lb, mid-1);
    	else
    		return binarySearch(nums, target, mid+1, ub);
    }
    
    
    public int canCompleteCircuit(int[] gas, int[] cost) {
    	if (gas==null || cost==null)
    		return -1;
    	int maxSeg = Integer.MIN_VALUE;
    	int maxInd = 0;
    	int total = 0;
    	int runningSum = 0;
    	int runningInd = 0;
    	
    	for (int i = 0; i< gas.length; i++) {
    		total += (gas[i] - cost[i]);
    		runningSum += (gas[i] - cost[i]);
    		//First we want to judge that the running sum is positive
    		//otherwise we will recompute the running sum
    		if (runningSum < 0) {
    			runningSum = 0;
    			runningInd = i+1;
    			continue;
    		}
    		if (runningSum > maxSeg) {
    			maxSeg = runningSum;
    			maxInd = runningInd;
    		}
    	}
    	if (total<0)
    		return -1;
    	for (int i = 0; i<gas.length; i++) {
    		runningSum += (gas[i] - cost[i]);
    		if (runningSum >maxSeg) {
    			maxSeg = runningSum;
    			maxInd = runningInd;
    		}
    		if (runningSum<0)
    			break;
    	}
    	return maxInd;
    }
    
    public int maximumGap(int[] nums) {
        int[] sortedResult = new int[nums.length];
        for (int i = 0; i<32; i++) {
        	int numZeros = 0;
        	for (int num:nums) {
        		int digit = (num&(1<<i))>>i;
        		if (digit ==0)
        			numZeros++;
        	}
        	int zero = 0;
        	int one = numZeros;
        	for (int num:nums) {
        		int digit = (num&(1<<i))>>i;
        		if (digit ==0)
        			sortedResult[zero++] = num;
        		else
        			sortedResult[one++] = num;
        	}
        	for (int j=0; j<nums.length; j++){
        		nums[j] = sortedResult[j];
        	}      		
        }
        int gap = 0;
        for (int i =1; i<nums.length; i++) {
        	gap = Math.max(nums[i] - nums[i-1], gap);
        }
        return gap;
    }
    
    public int findMiddle (int A, int B, int C) {
    	if ((A-B)*(C-A)>=0)
    		return A;
    	else if ((B-A)*(C-B)>=0)
    		return B;
    	return C;
    }
    
    public int trap(int[] height) {
    	if (height.length<=1)
    		return 0;
        int[] maxLeft = new int[height.length];
        int max = height[0];
        for (int i = 1; i<height.length; i++) {
        	max = Math.max(height[i-1], max);
        	maxLeft[i] = max;
        }
        int water = 0;
        max = height[height.length-1];
        for (int i = height.length-2; i>=0; i--) {
        	max = Math.max(height[i+1], max);
        	if (height[i]<max && height[i] < maxLeft[i]) {
        		water += Math.min(max, maxLeft[i]) - height[i];
        	}
        }
        return water;
    }
    
    public int[] twoSum(int[] nums, int target) {
    	if(nums==null || nums.length<2){
    		return null;
    	}
    	Arrays.sort(nums);
    	int i = 0;
    	int j = nums.length-1;
    	while (i<j){
    		int sum = nums[i] +nums[j];
    		if (sum == target){
    			return new int[]{i, j};
    		}
    		while (sum<target && i<j){
    			i++;
    		}
    		while (sum>target && i<j ){
    			j--;
    		}
    	}
    	if (nums[i] + nums[j] ==target){
    		return new int[]{i, j};
    	}
    	else{
    		return null;
    	}
    }
    
    public int[] productExceptSelf(int[] nums) {
    	if (nums==null || nums.length<1)
    		return nums;
        int p = 1;
        int[] ret = new int[nums.length];
        ret [0] = 1;
        for (int i = 1; i<nums.length; i++){
        	p *= nums[i-1];
        	ret[i] = p;
        }
        p = 1;
        for (int i = nums.length-2; i>=0; i--){
        	p *= nums[i+1];
        	ret[i] *= p;
        }
        return ret;
    }
    
    public String getHint(String secret, String guess) {
        int[] count = new int[128];
        int cow = 0;
        for (int i=0; i<secret.length(); i++) {
        	if (secret.charAt(i) == guess.charAt(i))
        		cow++;
        	else 
        		count[secret.charAt(i)]++;
        }
        int bull = 0;
        for (int i=0; i<guess.length(); i++) {
        	if (secret.charAt(i) != guess.charAt(i) && count[guess.charAt(i)]>0){
        		count[guess.charAt(i)]--;
        		bull++;
        	}
        }
        String ret = cow + "A" + bull + "B";
        return ret;
    }
    
    public List<List<Integer>> fourSum(int[] nums, int target) {
    	List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (nums==null || nums.length<4)
        	return result;
        Arrays.sort(nums);
        for (int i = 0; i< nums.length-3; i++){
        	if (i!=0 && nums[i] == nums [i-1])
        		continue;
        	for (int z = i+1; z<nums.length-2; z++){
        		if (z!=i+1&&nums[z]==nums [z-1])
        			continue;
        		int sum = target-nums[i]-nums[z];
        		
        		int j = z+1;
            	int k = nums.length-1;
            	while (j<k){
            		if (nums[j] + nums[k] == sum){
            			ArrayList<Integer> list =new ArrayList<> ();
            			list.add(nums[i]);
            			list.add(nums[z]);
            			list.add(nums[j]);
            			list.add(nums[k]);
            			result.add(list);
            			j++;
            			k--;
            			while (j<k && nums[j] == nums[j-1])
            				j++;
            			while (j<k && nums[k] == nums[k+1])
            				k--;
            		}
            		if (nums[j] + nums[k]<sum){
            			j++;
            		}
            		if (nums[j] + nums[k]>sum){
            			k--;
            		}
            		
            	}
        		
        	}
        	
        }
        return result;
    }
    
    
    public void wiggleSort(int[] nums) {
    	if (nums==null || nums.length==0)
    		return;
    	for (int i=0; i<nums.length-1; i++){
    		if (i%2==0){
    			if (nums[i]>nums[i+1])
    				swap(nums, i, i+1);
    		}
    		else{
    			if (nums[i]<nums[i+1])
    				swap(nums, i, i+1);
    		}
    	}
    }
    
    public void swap(int[] nums, int i, int j){
    	int temp = nums[i];
    	nums[i] = nums[j];
    	nums[j] = temp;
    }
    
    
    public List<List<Integer>> threeSum(int[] nums) {
    	List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (nums==null || nums.length<3)
        	return result;
        Arrays.sort(nums);
        for (int i = 0; i< nums.length-2; i++){
        	if (i!=0 && nums[i] == nums [i-1])
        		continue;
        	int sum = 0- nums[i];
        	int j = i+1;
        	int k = nums.length-1;
        	while (j<k){
        		if (nums[j] + nums[k] == sum){
        			ArrayList<Integer> list =new ArrayList<> ();
        			list.add(nums[i]);
        			list.add(nums[j]);
        			list.add(nums[k]);
        			result.add(list);
        			j++;
        			k--;
        			while (j<k && nums[j] == nums[j-1])
        				j++;
        			while (j<k && nums[k] == nums[k+1])
        				k--;
        		}
        		if (nums[j] + nums[k]<sum){
        			j++;
        		}
        		if (nums[j] + nums[k]>sum){
        			k--;
        		}
        		
        	}
        }
        return result;
    }
    
    public int[] twoSumHashSet(int[] nums, int target){
    	if(nums==null || nums.length<2){
    		return null;
    	}
    	HashMap<Integer, Integer> map = new HashMap<>();
    	for (int i = 0; i<nums.length; i++){
    		map.put(nums[i], i);
    	}
    	for (int i = 0; i<nums.length; i++){
    		if (map.containsKey(target- nums[i]) && i!=map.get(target-nums[i])){
    			return new int[]{i, map.get(target-nums[i])};
    		}
    	}
    	return null;
    	
    }
    
    public List<List<Integer>> palindromePairs(String[] words) {
    	HashMap<String, Integer> map = new HashMap<String, Integer>();
    	int nullInd = -1;
    	for (int i = 0; i<words.length; i++){
    		map.put(words[i], i);
    		if (words[i].length()==0){
    			nullInd = i;
    		}
    	}
    	List<List<Integer>> ret = new ArrayList<>();
    	for (int i = 0; i<words.length; i++){
    		String word = words[i];
    		for (int j =1; j<word.length(); j++) {
    			String subWord = new StringBuilder(word.substring(0, j)).reverse().toString();
    			if (map.containsKey(subWord) && isPalindrome(word+subWord)) {
    				ArrayList<Integer> array = new ArrayList<> ();
					array.add(i);
					array.add(map.get(subWord));
					ret.add(array);
    			}
    		}
    		String subWord = new StringBuilder(word).reverse().toString();
    		if (map.containsKey(subWord) && i!= map.get(subWord)) {
    			ArrayList<Integer> array = new ArrayList<> ();
				array.add(i);
				array.add(map.get(subWord));
				ret.add(array);
    		}
    		
    		for (int j = word.length()-1; j>=1; j--) {
    			subWord = new StringBuilder(word.substring(j)).reverse().toString();
    			if (map.containsKey(subWord) && isPalindrome (subWord+word)) {
    				ArrayList<Integer> array = new ArrayList<> ();
    				array.add(map.get(subWord));
					array.add(i);
					ret.add(array);
    			}
    		}
    		
    		if (nullInd!=-1 &&isPalindrome(word)) {
    			ArrayList<Integer> array1 = new ArrayList<> ();
				array1.add(nullInd);
				array1.add(i);
				ret.add(array1);
				
				ArrayList<Integer> array2 = new ArrayList<> ();
				array2.add(i);
				array2.add(nullInd);
				ret.add(array2);
    		}
    	}
    	return ret;
    }
    
    boolean isPalindrome(String str) {    
        int n = str.length();
        if (n==0)
        	return false;
        for( int i = 0; i < n/2; i++ )
            if (str.charAt(i) != str.charAt(n-i-1)) return false;
        return true;    
    }
    
    public int findMin(int[] nums, int start, int end){
    	if (end-start<=1){
    		return Math.min(nums[start], nums[end]);
    	}
    	int mid = start + (end-start)/2;
    	if (nums[mid]>nums[end])
    		return findMin(nums, mid, end);
    	else
    		return findMin(nums, start, mid);
    }
    
    public int maxArea(int[] height) {
    	int i=0;
    	int j=height.length-1;
    	int wall = Math.min(height[i],  height[j]);
    	int maxArea = wall*(j-i);
    	boolean isLeft = (height[i]==wall);
    	while (i!=j){
    		if (isLeft)
    			i++;
    		else
    			j--;
    		wall = Math.min(height[i],  height[j]);
			isLeft = (height[i]==wall);
    		int area = wall*(j-i);
			maxArea = Math.max(maxArea, area);
    	}
        return maxArea;
    }
    
    
    public boolean canJump(int[] nums) {
        int maxRange = 0;
        if (nums==null||nums.length<=1){
        	return true;
        }
        for (int i=0; i<nums.length; i++){
        	if (i>maxRange)
        		return false;
        	maxRange = Math.max(maxRange, i+nums[i]);
        }
        return true;
    }
    
    public int missingNumber(int[] nums) {
    	int size = nums.length;
    	int sum = (size+1)*size/2;
    	for (int num:nums){
    		sum -= num;
    	}
    	return sum;
    }
    
    public int searchInsert(int[] nums, int target) {
    	if (nums==null || nums.length==0)
    		return 0;
        if (target>nums[nums.length-1])
        	return nums.length;
        return binarySearch(nums, target, 0, nums.length);
    }
    
//    public int binarySearch(int[] nums, int target, int start, int end){
//    	if (start==end){
//    		return start;
//    	}
//    	int mid = start + (end-start)/2;
//    	if (nums[mid]>=target)
//    		return binarySearch(nums, target, start, mid);
//    	else
//    		return binarySearch(nums, target, mid+1, end);
//    }
    public boolean isValid(String s) {
    	if (s==null ||s.length()==0)
    		return false;
        Stack<Character> stack = new Stack<>();
       for (int i = 0; i<s.length(); i++) {
    	   char c = s.charAt(i);
    	   if (c=='(' || c=='[' || c=='{') {
    		   stack.push(c);
    	   }
    	   else if (c==')' || c==']' || c=='}') {
    		   if (stack.isEmpty())
    			   return false;
    		   char counter = stack.pop();
    		   if (c==')' && counter!='(')
    			   return false;
    		   if (c=='}' && counter!='{')
    			   return false;
    		   if (c==']' && counter!='[')
    			   return false;
    	   }
       }
       return stack.isEmpty();
    }
    
    public boolean increasingTriplet(int[] nums) {
    	if (nums==null || nums.length<3)
    		return false;
        int[] valley = new int[nums.length];
        valley[0] = Integer.MAX_VALUE;
        valley[nums.length-1] = Integer.MIN_VALUE;
        for (int i = 1; i<nums.length-1; i++){
        	valley[i] = Math.min(valley[i-1], nums[i-1]);
        }
        for (int i = nums.length-2; i>0;  i--){
        	int rightMax = Math.max(valley[i+1], nums[i+1]);
        	if (nums[i]>valley[i] && nums[i] < rightMax)
        		return true;
        	valley[i] = rightMax;
        }
        return false;
    }
    
    public int minPatches(int[] nums, int n) {
    	if (n==0)
    		return 0;
    	long miss = 1;
    	int index = 0;
    	int add = 0;
    	while (miss<=n) {
    		if (index < nums.length) {
    			if (miss > nums[index]) {
    				miss += nums[index++];
    			}
    			else {
    				miss = miss<<1;
    				add ++;
    			}
    		}
    		else {
    			miss = miss<<1;
    			add++;
    		}
    	}
    	return add;
    }
    
    
    public int maxEnvelopes(int[][] envelopes) {
    	if (envelopes == null || envelopes.length == 0)
    		return 0;
        Arrays.sort(envelopes, new Comparator<int[]>(){
			@Override
			public int compare(int[] o1, int[] o2) {
				if (o1[0] != o2[0])
					return o1[0] - o2[0];
				else
					return o2[1] - o1[1];
			}
        });
        int max = 1;
        int[] maxLength = new int[envelopes.length];
        for (int i = 0; i < envelopes.length; i++) {
        	maxLength[i] = 1;
        	for (int j = 0; j < i; j++) {
        		if (envelopes[i][0] > envelopes[j][0] 
        				&& envelopes[i][1] > envelopes[j][1]) {
        			maxLength[i] = Math.max(maxLength[i], maxLength[j] + 1);
        		}
        	}
        	max = Math.max(max, maxLength[i]);
        }
        return max;
    }
    
    public int[] getModifiedArray(int length, int[][] updates) {
    	int[] result = new int[length];
    	for (int[] update: updates) {
    		result[update[0]] += update[2];
    		if (update[1] < length - 1)
    			result[update[1] + 1] -= update[2];
    	}
    	for (int i = 1; i < length; i ++) {
    		result[i] += result[i - 1];
    	}
    	return result;
    }
    
    public int longestConsecutive(int[] nums) {
    	// a number n should point at its successor n+1
        int[] union = new int[nums.length];
        // a map to store value--indices
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i< nums.length; i++) {
        	union[i] = -1;
        	if (map.containsKey(nums[i]+1)) {
        		union[i] = map.get(nums[i] + 1);
        	}
        	if (map.containsKey(nums[i] -1)){
        		union[map.get(nums[i] -1)] = i;
        	}
        	map.put(nums[i], i);
        }
        //now the union has been established,
        //we want to run one pass to find the longest subsequence
        int[] distance = new int[nums.length];
        int maxDist = 1;
        for (int i = 0; i<nums.length; i++) {
        	if (union[i]!=-1 && distance[i] ==0){
        		maxDist = Math.max(findDist(union, distance, i), maxDist);
        	}
        }
        return maxDist;
    }
    
    private int findDist (int[] union, int[] dist, int i){
    	if (union[i] == -1){
    		dist[i] = 1;
    		return 1;
    	}
    	if (dist[i]!=0)
    		return dist[i];
    	else
    		dist[i] = findDist(union, dist, union[i]) +1;
    	return dist[i];
    }
    
    public int findDuplicate(int[] nums) {
    	int fast = 0;
    	int slow = 0;
    	do{
    		fast = nums[nums[fast]];
    		slow = nums[slow];
    	} while (fast!=slow);
    	fast = 0;
    	while (fast!=slow) {
    		fast = nums[fast];
    		slow = nums[slow];
    	}
    	return fast;
    }
    
    public List<String> findRepeatedDnaSequences(String s) {
    	HashMap<String, Integer> table = new HashMap<>();
    	List<String> list = new ArrayList<> ();
    	if (s==null ||s.length()<=10)
    		return list;
    	
    	for (int i = 0; i<=s.length()-10; i++){
    		String subS = s.substring(i, i+10);
    		if (table.containsKey(subS) && table.get(subS)==1){
    			list.add(subS);
    			table.put(subS, 2);
    			continue;
    		}
    		if (!table.containsKey(subS))
    			table.put(subS, 1);
    	}
    	return list;
    }
    
    public static int KMPsearch (String haystack, String needle) {
    	if (haystack==null || needle== null)
            return -1;
        if (needle.length() ==0)
            return 0;
    	int j = 0;
    	int[] P = new int[needle.length()];
    	for (int i = 1; i<needle.length(); i++){
    		while (j!=0 && needle.charAt(i)!=needle.charAt(j))
    			j = P[j-1];
    		if (needle.charAt(i)==needle.charAt(j))
    			j++;
    		P[i] = j;
    	}
    	j = 0;
    	for (int i = 0; i<haystack.length(); i++) {
    		while (j!=0 && haystack.charAt(i)!=needle.charAt(j)){
    			j = P[j-1];
    		}
    		if (haystack.charAt(i) == needle.charAt(j)) {
    			if (j==needle.length())
    				return i-j;
    			j++;
    		}
    		
    	}
    	return -1;
    }
    
    
    public static void printTable (int[][] table) {
    	for (int i = 0; i<table.length; i++) {
    		for (int j = 0; j<table[0].length; j++){
    			System.out.print(table[i][j] +" ");
    		}
    		System.out.println("");
    	}
    }
    
    public int minTotalDistance(int[][] grid) {
    	ArrayList<Integer> Xaxis = new ArrayList<> ();
    	ArrayList<Integer> Yaxis = new ArrayList<> ();
    	for (int i = 0; i < grid.length; i++) {
    		for (int j = 0; j < grid[0].length; j++) {
    			if (grid[i][j] == 1) {
    				Xaxis.add(i);
    				Yaxis.add(j);
    			}
    		}
    	}
    	Collections.sort(Xaxis);
    	Collections.sort(Yaxis);
    	int centerX = 0;
    	int centerY = 0;
    	if (Xaxis.size() % 2 == 1) {
    		centerX = Xaxis.get(Xaxis.size() / 2);
    		centerY = Yaxis.get(Yaxis.size() / 2);
    	}
    	else {
    		centerX = (Xaxis.get(Xaxis.size() / 2 - 1) + Xaxis.get(Xaxis.size() / 2)) / 2;
    		centerY = (Yaxis.get(Yaxis.size() / 2 - 1) + Yaxis.get(Yaxis.size() / 2)) / 2;
    	}
    	int dist = 0;
    	for (int i = 0; i < grid.length; i++) {
    		for (int j = 0; j < grid[0].length; j++) {
    			if (grid[i][j] == 1) {
    				dist += (Math.abs(i - centerX) + Math.abs(j - centerY));
    			}
    		}
    	}
    	return dist;
    }
    
    
    
    public static void main(String[] args){
    	Solutions s = new Solutions();
    	int[][] patch = new int[][] {{46,89},{50,53},{52,68},{72,45},{77,81}};
    	System.out.println(s.maxEnvelopes(patch));
    }

}
