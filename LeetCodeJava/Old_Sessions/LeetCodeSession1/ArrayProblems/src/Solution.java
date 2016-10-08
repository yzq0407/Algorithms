import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Stack;
import java.util.Comparator;
import java.util.Collections;


public class Solution {
    public int findMin(int[] nums) {
    	if (nums==null||nums.length==0)
    		return -1;
        return findMin(nums, 0, nums.length-1);
    }
    
    public boolean isAdditiveNumber(String num) {
        char[] num_arr = num.toCharArray();
        for (int i = 1; i < num_arr.length - 1; i ++) {
        	for (int j = i + 1; j < num_arr.length; j ++) {
        		if (j > i + 1 && num_arr[i] == '0')
        			break;
        		if (isAdditiveNumberHelper(num_arr, 0, i, j))
        			return true;
        	}
        }
        return false;
    }
    
    public boolean isAdditiveNumberHelper(char[] num, int i, int j, int k) {
    	if (k == num.length)
    		return true;
    	int p1 = j - 1;
    	int p2 = k - 1;
    	StringBuilder sb = new StringBuilder();
    	int carry = 0;
    	while (p1 >= i && p2 >= j) {
    		int sum = (num[p1--] - '0') + (num[p2--] - '0') + carry;
    		carry = sum / 10;
    		sum = sum % 10;
    		sb.insert(0, sum);
    	}
    	while (p1 >= i) {
    		int sum = (num[p1--] - '0') + carry;
    		carry = sum / 10;
    		sum = sum % 10;
    		sb.insert(0, sum);
    	}
    	while (p2 >= j) {
    		int sum = (num[p2--] - '0') + carry;
    		carry = sum / 10;
    		sum = sum % 10;
    		sb.insert(0, sum);
    	}
    	if (carry == 1) {
    		sb.insert(0, carry);
    	}
    	char[] sum = sb.toString().toCharArray();
    	p1 = k;
    	p2 = 0;
    	while (p1 < num.length && p2 < sum.length) {
    		if (sum[p2++] != num[p1++])
    			return false;
    	}
    	if (p1 == num.length && p2 != sum.length)
    		return false;
    	return isAdditiveNumberHelper(num, j, k, p1);
    }
    
    public List<List<String>> groupAnagrams(String[] strs) {
    	int[] primes = getPrimes(26);
        HashMap<Long, List<String>> map = new HashMap<> ();
        for (String str : strs) {
        	long hashCode = hash(str, primes);
        	List<String> anagrams;
        	if (map.containsKey(hashCode)) 
        		anagrams = map.get(hashCode);
        	else
        		anagrams = new ArrayList<String> ();
        	anagrams.add(str);
        	map.put(hashCode, anagrams);
        }
        List<List<String>> retAnagrams = new ArrayList<> ();
        for (Long key: map.keySet()){
        	List<String> anagrams = map.get(key);
        	Collections.sort(anagrams);
        	retAnagrams.add(anagrams);
        }
        return retAnagrams;
    }
    
    public long hash(String str, int[] primes) {
    	long code = 1;
    	char[] charArr = str.toCharArray();
    	for (int i = 0; i < charArr.length; i++) {
    		code *= primes[charArr[i] - 'a'];
    	}
    	return code;
    }
    
    public int[] getPrimes(int n){
    	int[] primes = new int[n];
    	primes[0] = 2;
    	int current = 3, i = 1;
    	while (i < n) {
    		boolean isPrime = true;
    		for (int j = 0; primes[j] <= current / primes[j]; j ++) {
    			if (current % primes[j] == 0) {
    				isPrime = false;
    				break;
    			}
    		}
    		if (isPrime) {
    			primes[i++] = current;
    		}
    		current++;
    	}
    	return primes;
    }
    
    public int lengthOfLIS(int[] nums) {
    	int maxSequence = 1;
    	int sequence = 1;
    	for (int i = 1; i<nums.length; i++) {
    		int pos = binarySearch(nums, nums[i], 0, maxSequence-1);
    		nums[pos] = nums[i];
    		maxSequence = Math.max(maxSequence, pos+1);
    	}
        return maxSequence;
    }
    
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
    
    public int binarySearch (int[] nums, int target, int lb, int ub) {
    	if (target>nums[ub])
    		return ub+1;
    	while (ub > lb+1) {
    		int mid = lb + (ub - lb)/2;
    		if (nums[mid] > target){
    			ub = mid - 1;
    			continue;
    		}
    		else {
    			lb = mid;
    			continue;
    		}
    	}
    	if (target>nums[ub])
    		return ub+1;
    	else if (target>nums[lb])
    		return lb+1;
    	else
    		return lb;
    }
    
    public int sum (int[] nums, int d){
    	int sum = 0;
    	while (d>0){
    		//System.out.println(d);
    		sum += nums[d];
    		d -= (d&-d);
    	}
    	return sum;
    }
    
    public void add (int[] nums, int a, int d){
    	while (d<nums.length) {
    		nums[d] += a;
    		d += (d&-d);
    	}
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
    
    
    public static void main(String[] args){
    	Solution s = new Solution();
    	int[] test = new int[] {1,3,6,7,9,4,10,5,6};
    	int[] sum = new int[10];
    	
    	System.out.println(s.isAdditiveNumber("101"));
    	//    	for (List<Integer> list: s.fourSum(test, -236727523)){
//    		for (int i:list){
//    			System.out.print(i+"-->");
//    		}
//    		System.out.println();
//    	}
    }

}
