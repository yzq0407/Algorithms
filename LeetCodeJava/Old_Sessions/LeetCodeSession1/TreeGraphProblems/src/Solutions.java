import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.LinkedList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Comparator;


public class Solutions{
	
	//for the interval questions
	public static class Interval {
			int start;
			int end;
			Interval() { start = 0; end = 0; }
			Interval(int s, int e) { start = s; end = e; }
			
			public String toString(){
				return start + "-->" +end;
			}
	}
	
	public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null)
        	return root;
        TreeNode ret = root;
        while (ret.left != null)
        	ret = ret.left;
        upsideDownBinaryTreeHelper(root);
        return ret;
    }
	
	int order;
    public String alienOrder(String[] words) {
        @SuppressWarnings("unchecked")
		HashMap<Character, Set<Character>> map = new HashMap<>();
        setupMap(map, words, 0, words.length - 1, 0);
        order = 1;
        HashMap<Character, Integer> isVisited = new HashMap<>();
        for (Character c : map.keySet()) {
        	if (!topologicalSort(map, isVisited, c))
        		return "";
        }
        char[] ret = new char[isVisited.size()];
        for (Character c: isVisited.keySet()) {
        	ret[ret.length - isVisited.get(c)] = c;
        }
        return String.valueOf(ret);
    }
	
	public boolean topologicalSort(HashMap<Character, Set<Character>> map, 
			HashMap<Character, Integer> isVisited,
			char current) {
		if (isVisited.containsKey(current) && isVisited.get(current) == -1)
			return false;
		if (isVisited.containsKey(current)) {
			return true;
		}
		if (!map.containsKey(current) || map.get(current).size() == 0) {
			isVisited.put(current, order++);
			return true;
		}
		isVisited.put(current, -1); 
		for (char des : map.get(current)) {
			if (!topologicalSort(map, isVisited, des))
				return false;
		}
		isVisited.put(current, order++); 
		return true;
		
	}
	
	public void setupMap(HashMap<Character, Set<Character>> map,
			String[] words, int start, int end, int digit) {
		if(start > end)
			return;
		char prev = '#';
		int subStart = start;
		int subEnd = start;
		for (int i = start; i <= end; i++) {
			String word = words[i];
			//these words has the same prefix, now compare the digit's char
			if (digit >= word.length())
				continue;
			char c = word.charAt(digit);
			if (!map.containsKey(c)) {
				map.put(c, new HashSet<Character> ());
			}
			if (prev == '#') {
				prev = c;
				continue;
			}
			if (prev == c) {
				subEnd = i;
			}
			if (prev != c) {
				if (map.containsKey(prev)) {
					map.get(prev).add(c);
				}
				else {
					Set<Character> set = new HashSet<>();
					set.add(c);
					map.put(prev, set);
				}
				setupMap(map, words, subStart, subEnd, digit + 1);
				subStart = i;
				subEnd = i;
			}
			prev = c;
		}
		if (prev == '#')
			return;
		setupMap(map, words, subStart, subEnd, digit + 1);
	}

	
	
	public int longestConsecutive(TreeNode root) {
		return longestConsecutiveHelper(root)[0];
    }
	
	public int[] longestConsecutiveHelper(TreeNode root) {
		if (root == null) {
			return new int[] {0, 0};
		}
		int[] left = longestConsecutiveHelper(root.left);
		int[] right = longestConsecutiveHelper(root.right);
		int longestInclude = 1;
		int longestPreclude = Math.max(left[0], right[0]);
		if (root.left != null && root.left.val == root.val + 1) {
			if (left[1] == 1)
				longestInclude = Math.max(left[0] + 1, longestInclude);
			else
				longestInclude = Math.max(2, longestInclude);
		}
		if (root.right != null && root.right.val == root.val + 1){
			if (right[1] == 1)
				longestInclude = Math.max(right[0] + 1, longestInclude);
			else
				longestInclude = Math.max(2, longestInclude);
		}
		int isInclude = 0;
		if (longestInclude > longestPreclude)
			isInclude = 1;
		return new int[]{Math.max(longestInclude, longestPreclude), isInclude};
	}
	

	public interface NestedInteger {

		public boolean isInteger();

    	public Integer getInteger();
    	public List<NestedInteger> getList();
	}
	 
	 
	public int depthSumInverse(List<NestedInteger> nestedList) {
        int maxDepth = findMaxDepth(nestedList, 1);
        return sumNestedInteger(nestedList, 0, maxDepth);
    }
	
	public int findMaxDepth(List<NestedInteger> list, int currentDepth) {
		int max = currentDepth;
		if (list.size() == 0)
			return currentDepth - 1;
		for (NestedInteger ni: list) {
			if (!ni.isInteger()) {
				max = Math.max(max, findMaxDepth(ni.getList(), currentDepth + 1));
			}
		}
		return max;
	}
	
	public int sumNestedInteger(List<NestedInteger> list, int currentDepth, int maxDepth){
		int sum = 0;
		for (NestedInteger ni: list) {
			if (ni.isInteger()) {
				sum += (maxDepth - currentDepth) * ni.getInteger();
			}
			else {
				sum += sumNestedInteger(ni.getList(), currentDepth + 1, maxDepth);
			}
		}
		return sum;
	}
	
	
	public TreeNode upsideDownBinaryTreeHelper(TreeNode root) {
		if (root == null)
        	return root;
		TreeNode left = root.left, right = root.right;
		root.left = null;
		root.right = null;
		while (left != null) {
			TreeNode tempL = left.left;
			TreeNode tempR = right.right;
			left.left = right;
			left.right = root;
			root = left;
			left = tempL;
			right = tempR;
		}
		return root;
	}
	
	public List<List<String>> findLadders(String beginWord, String endWord, Set<String> wordList) {
        HashMap<String, List<String>> map = new HashMap<> ();
        wordList.add(endWord);
        boolean found = false;
        List<List<String>> ret = new ArrayList<>();
        if (wordList.isEmpty())
        	return ret;
        Queue<String> queue = new LinkedList<String>();
        map.put(beginWord, new ArrayList<String>());
        queue.offer(beginWord);
        while (!found) {
        	int size = queue.size();
        	//do it layer by layer
        	HashSet<String> layer = new HashSet<>();
        	for (int i = 0; i < size; i++) {
        		String sNode = queue.poll();
        		char[] s = sNode.toCharArray();
        		for (int j = 0; j < s.length; j ++) {
        			char letter = s[j];
        			for (int c = 0; c < 26; c ++) {
        				if (c == letter - 'a')
        					continue;
        				s[j] = (char) ('a' + c);
        				String neighbor = String.valueOf(s);
        				if (wordList.contains(neighbor)) {
        					if (layer.contains(neighbor)) {
        						map.get(neighbor).add(sNode);
        					}
        					else if (!map.containsKey(neighbor)) {
        						queue.offer(neighbor);
        						List<String> adj = new ArrayList<> ();
        						adj.add(sNode);
        						map.put(neighbor, adj);
        						layer.add(neighbor);
        					}
        				}
        			}
        			s[j] = letter;
        		}
        	}
        	if (layer.contains(endWord))
        		found = true;
        	if (queue.size() == 0)
        		return ret;
        }
        dfs(ret, beginWord, map, endWord, new LinkedList<String>());
        return ret;
    }
	
	public void dfs (List<List<String>> ret, String startWord,
			Map<String, List<String>> map, String current, LinkedList<String> list){
		if (current.equals(startWord)) {
			ArrayList<String> aPath = new ArrayList<> ();
			aPath.add(startWord);
			aPath.addAll(list);
			ret.add(aPath);
			return;
		}
		else {
			list.addFirst(current);;
			for (String neighbor: map.get(current)){
				dfs(ret, startWord, map, neighbor, list);
			}
			list.removeFirst();
		}
	}
	
	public boolean verifyPreorder(int[] preorder) {
        Stack<Integer> s = new Stack<> ();
        int min = Integer.MIN_VALUE;
        int j = -1;
        for (int i = 0; i < preorder.length; i ++) {
        	if (preorder[i] < min)
        		return false;
        	while (j >= 0 && preorder[j] < preorder[i]) {
        		min = preorder[j--];
        	}
        	preorder[++j] = preorder[i];
        }
        return true;
    }
	
	public List<Integer>  closestKValues(TreeNode root, double target, int k) {
		Stack<TreeNode> predecessors = new Stack<> ();
		Stack<TreeNode> successors = new Stack<> ();
		PriorityQueue<Integer> values = new PriorityQueue<>(new Comparator<Integer>(){
			@Override
			public int compare(Integer o1, Integer o2) {
				return Double.compare(Math.abs(o1 - target), Math.abs(o2 - target)); 
			}
		});
		PriorityQueue<Integer> values2 = new PriorityQueue<>(new Comparator<Integer>(){
			@Override
			public int compare(Integer o1, Integer o2) {
				return Double.compare(Math.abs(o1 - target), Math.abs(o2 - target)); 
			}
		});
		TreeNode pred = root;
		TreeNode succ = root;
		while (pred != null) {
			if (pred.val > target) {
				pred = pred.left;
			}
			else {
				predecessors.push(pred);
				pred = pred.right;
			}
		}
		while (succ != null) {
			if (succ.val <= target) {
				succ = succ.right;
			}
			else {
				successors.push(succ);
				succ = succ.left;
			}
		}
		//Do in order traverse(reverse) of the predessors
		while (!predecessors.isEmpty() && values.size() < k) {
			TreeNode node = predecessors.pop();
			values.add(node.val);
			node = node.left;
			Stack<TreeNode> inOrderTraverse = new Stack<> ();
			while ((node != null || !inOrderTraverse.isEmpty()) && values.size() < k) {
				if (node == null) {
					node = inOrderTraverse.pop();
					values.add(node.val);
					node = node.left;
				}
				else {
					inOrderTraverse.push(node);
					node = node.right;
				}
			}
		}
		//Do in order traverse of the successors
		while (!successors.isEmpty() && values2.size() < k) {
			TreeNode node = successors.pop();
			values2.add(node.val);
			node = node.right;
			Stack<TreeNode> inOrderTraverse = new Stack<> ();
			while ((node != null || !inOrderTraverse.isEmpty()) && values2.size() < k) {
				if (node == null) {
					node = inOrderTraverse.pop();
					values2.add(node.val);
					node = node.right;
				}
				else {
					inOrderTraverse.push(node);
					node = node.left;
				}
			}
		}
		//combine two priority queues 
		List<Integer> list = new ArrayList<> ();
		while (values.size() != 0 && values2.size() != 0 && list.size() < k) {
			if (Math.abs((long)values.peek() - (long)target) < Math.abs((long)values2.peek() - (long)target))
				list.add(values.poll());
			else
				list.add(values2.poll());
		}
		while (list.size() < k) {
			if (values.size() > 0)
				list.add(values.poll());
			else
				list.add(values2.poll());
		}
		return list;
    }
	
    
    
    public List<List<Integer>> getFactors(int n) {
    	List<List<Integer>> ret = getFactors(n, 2);
    	ret.remove(ret.size() - 1);
    	return ret;
    }
    
    
    public int largestBSTSubtree(TreeNode root) {
        int[] tree = largestBSTSubTreeHelper(root);
        return tree[2];
    }
    
    public int[] largestBSTSubTreeHelper(TreeNode root) {
    	if (root == null)
    		return new int[] {Integer.MAX_VALUE, Integer.MIN_VALUE, 0, 1};
    	int[] left = largestBSTSubTreeHelper(root.left);
    	int[] right = largestBSTSubTreeHelper(root.right);
    	if (left[1] <= root.val && right[0] >= root.val && left[3] == 1 && right[3] ==1) {
    		int size = left[2] + right[2] + 1;
    		int min = root.left == null? root.val: left[0];
    		int max = root.right == null? root.val: right[1];
    		return new int[] {min, max, size, 1};
    	}
    	else {
    		return new int[] {0, 0, Math.max(left[2], right[2]), 0};
    	}
    }
    
    class Coordinate {
    	int x;
    	int y;
    	Coordinate(int x, int y) {
    		this.x = x;
    		this.y = y;
    	}
    }
    
    
    public void wallsAndGates(int[][] rooms) {
        for (int i = 0; i < rooms.length; i++) {
        	for (int j = 0; j < rooms[0].length; j++) {
        		if (rooms[i][j] == 0) {
        			bfs(rooms, i, j);
        		}
        	}
        }
    }
    
    
    
    public void bfs (int[][] rooms, int i, int j) {
    	int[] dx = new int[]{-1, 1, 0, 0};
    	int[] dy = new int[]{0, 0, -1, 1};
    	Queue<Coordinate> q = new LinkedList<> ();
    	q.offer(new Coordinate(i, j));
    	int dist = 1;
    	while (!q.isEmpty()) {
    		int size = q.size();
    		for (int count = 0; count < size; count++) {
    			Coordinate coord = q.poll();
    			int x = coord.x;
    			int y = coord.y;
    			for (int direc = 0; direc < 4; direc++) {
    				int nx = x + dx[direc];
    				int ny = y + dy[direc];
    				if (nx >= 0 && nx < rooms.length && ny >=0 && ny < rooms[0].length
    						 && rooms[nx][ny] > dist) {
    					q.offer(new Coordinate(nx, ny));
    					rooms[nx][ny] = dist;
    				}
    			}
    		}
    		dist++;
    	}
    }
    
    
    public boolean canAttendMeetings(Interval[] intervals) {
        int prevEnd = Integer.MIN_VALUE;
        Arrays.sort(intervals, new Comparator<Interval>(){
			@Override
			public int compare(Interval o1, Interval o2) {
				// TODO Auto-generated method stub
				return o1.start - o2.start;
			}
        	
        });
        for (Interval interval: intervals) {
        	if (interval.start < prevEnd)
        		return false;
        	prevEnd = interval.end;
        }
        return true;
    }
    
   
    
    
    
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if (root == null || p == null)
        	return null;
        if (root.val <= p.val)
        	return inorderSuccessor(root.right, p);
        TreeNode node = inorderSuccessor(root.left, p);
        if (node != null)
        	return node;
        return root;
    }
    
    
    public List<List<Integer>> getFactors(int n, int min) {
    	List<List<Integer>> ret = new ArrayList<> ();
    	if (n == 1) {
    		List<Integer> solution = new ArrayList<> ();
    		ret.add(solution);
    		return ret;
    	}
    	for (int i = min; i <= n; i ++) {
    		if (n % i == 0) {
    			for (List<Integer> list : getFactors(n/i, i)) {
    				ArrayList<Integer> solution = new ArrayList<> ();
    				solution.add(i);
    				solution.addAll(list);
    				ret.add(solution);
    			}
    		}
    	}
    	return ret;
    }
	
	public List<List<String>> groupStrings(String[] strings) {
        HashMap<String, List<String>> map = new HashMap<>();
        for (String s : strings) {
        	String hash = hash(s);
        	List<String> list = map.containsKey(hash)? map.get(hash) : new ArrayList<>();
        	list.add(s);
        	map.put(hash, list);
        }
        List<List<String>> ret = new ArrayList<>();
        for (String key: map.keySet()) {
        	List<String> list = map.get(key);
        	Collections.sort(list);
        	ret.add(list);
        }
        return ret;
    }
	
	public String hash (String s) {
		StringBuilder sb = new StringBuilder();
		for (int i = 1; i < s.length(); i++) {
			int diff = s.charAt(i) - s.charAt(i - 1); 
			diff = diff < 0 ? diff + 26 : diff;
			sb.append(diff + "#");
		}
		return sb.toString();
	}
	
	public boolean verifyPreorder(int[] preorder, int start, int end) {
		if (start >= end - 1)
			return true;
		boolean leftFound = false;
		int rightStart = end + 1;
		for (int i = start + 1; i < end; i ++){
			if (preorder[i] > preorder[start] && !leftFound) {
				leftFound = true;
				rightStart = i;
				continue;
			}
			if (leftFound && preorder[i] <= preorder[start]) {
				return false;
			}
		}
		return verifyPreorder(preorder, start + 1, rightStart - 1) 
				&& verifyPreorder(preorder, rightStart, end);
	}
	
	
	
	public int nthSuperUglyNumber(int n, int[] primes) {
		int length = primes.length;
		int[] uglyPointers = new int[length];
		int[] vals = new int[n];
		vals[0]= 1;
		
		for (int i = 1; i<n; i++) {
			vals[i] = Integer.MAX_VALUE;
			for (int j = 0; j<primes.length; j++) {
				vals[i] = Math.min(vals[uglyPointers[j]]*primes[j] , vals[i]);
			}
			for (int j = 0; j<primes.length; j++) {
				if (vals[uglyPointers[j]]*primes[j] == vals[i]){
					uglyPointers[j]++;
				}
			}
		}
		return vals[n-1];
    }
	
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
    	//use post-order traverse to serialize
    	StringBuilder sb = new StringBuilder();
    	serialize(root, sb);
    	return sb.toString();
    }
    
    private void serialize(TreeNode root, StringBuilder sb) {
    	if (root == null) {
    		sb.append("x ");
    		return;
    	}
    	sb.append(root.val + " ");
    	serialize(root.left, sb);
    	serialize(root.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
    	String[] treeS = data.split(" ");
        Queue<String> q = new LinkedList<>();
        for (String s : treeS) {
        	q.add(s);
        }
        return deserialize(q);
        
    }
    
    public List<String> findItinerary (String[][] tickets) {
    	HashMap<String, PriorityQueue<String>> graph = new HashMap<>();
    	for (String[] ticket : tickets) {
    		String depart = ticket[0];
    		String arrive = ticket[1];
    		if (graph.containsKey(depart))
    			graph.get(depart).offer(arrive);
    		else {
    			PriorityQueue<String> set = new PriorityQueue<String>();
    			set.offer(arrive);
    			graph.put(depart, set);
    		}
    	}
    	List<String> itinerary = new ArrayList <> ();
    	String current = "JFK";
    	dfs(graph, current, itinerary);
    	return itinerary;
    	
    }
    
    
    
    private void dfs (HashMap<String, PriorityQueue<String>> graph, String current
    		, List<String> itinerary) {
		while (graph.containsKey(current) && !graph.get(current).isEmpty()) {
			dfs (graph, graph.get(current).poll(), itinerary);
		}
		itinerary.add(0, current);
    }
	
    private TreeNode deserialize(Queue<String> q) {
    	if (q.isEmpty())
    		return null;
    	String next = q.poll();
    	if (next.equals("x"))
    		return null;
    	TreeNode root = new TreeNode(Integer.parseInt(next));
    	root.left = deserialize(q);
    	root.right = deserialize(q);
    	return root;
    }
    
    
    public void rotate(int[] nums, int k) {
    	k = k%nums.length;
        if (k==0)
            return;
    	for (int offset = 0; offset<getGCD(nums.length, k); offset++) {
	        int next = getAppropPos(offset, nums.length, k);
	        while (next!=offset) {
	        	int temp = nums[offset];
	        	nums[offset] = nums[next];
	        	nums[next] = temp;
	        	next = getAppropPos(next, nums.length, k);
	        }
    	}
    }
    
    private int getGCD (int major, int minor) {
    	while (major%minor !=0) {
    		int temp = minor;
    		minor = major%minor;
    		major = temp;
    	}
    	return minor;
    }
    
    private int getAppropPos(int pos, int total, int offset) {
    	if (pos + offset< total)
    		return pos+offset;
    	else 
    		return pos+offset-total;
    }
    
    public String simplifyPath(String path) {
    	Stack<String> documents = new Stack<>();
    	int i = 0;
    	while (i<path.length()) {
    		if (path.charAt(i) == '/') {
    			i++;
    			continue;
    		}
    		int start = i;
    		while (i<path.length() && path.charAt(i)!='/') i++;
    		String document = path.substring(start, i);
    		if (document.equals("."))
    			continue;
    		if (document.equals("..")) {
    			if (!documents.isEmpty())
    			    documents.pop();
    			continue;
    		}	
    		documents.add(path.substring(start, i));
    	}
    	String ret = "";
    	if (documents.isEmpty()) {
    	    return "/";
    	}
    	while (!documents.isEmpty()) {
    		ret = "/" + documents.pop() + ret;
    	}
    	return ret;
    }
    
    public boolean isValidSerialization(String preorder) {
    	int slots = 2;
    	int i = 0;
    	if (preorder.charAt(0)=='#')
    		return preorder.length()==1;
    	String[] arr = preorder.split(",");
    	for (i = 1; i<arr.length; i++) {
    		if (arr[i].equals("#"))
    			slots--;
    		else 
    			slots++;
    		if (slots <=0 && i!=arr.length-1)
    			return false;
    	}
    	return slots==0;
    }
    
    
    
	public int rob(TreeNode root) {
        int[] scheme = robWithThisHouse(root);
        return Math.max(scheme[0], scheme[1]);
    }
	
	public int[] robWithThisHouse(TreeNode root){
		if (root==null)
			return new int[] {0, 0};
		int[] left = robWithThisHouse(root.left);
		int[] right = robWithThisHouse(root.right);
		int maxNotInclude = 0;
		for (int i = 0; i<2; i++){
			for (int j = 0; j<2; j++){
				maxNotInclude = Math.max(left[i] + right[j], maxNotInclude);
			}
		}
		return new int[]{root.val + left[1] + right[1], maxNotInclude};
	}
	

	public int countNodes(TreeNode root) {
		if (root==null)
			return 0;
		int count = 0;
		int l = leftHeight(root.left);
		int r = leftHeight(root.right);
		if (l==r) {
			count += (1<<l) + countNodes(root.right);
		}
		else {
			count += (1<<r) + countNodes(root.left);
		}
		return count;
	}
	
	public int leftHeight (TreeNode root){
		int height = 0;
		while(root!=null) {
			height++;
			root = root.left;
		}
		return height;
	}
	
	public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
		List<Interval> ret = new ArrayList<> ();
		int mergeSt = newInterval.start;
		int mergeEnd = newInterval.end;
		int interPos = 0;
		for (Interval interval: intervals) {
			if (interval.end<newInterval.start) {
				ret.add(interval);
				interPos++;
			}
			else if (interval.start>newInterval.end) {
				ret.add(interval);
			}
			else {
				mergeSt = Math.min(interval.start, mergeSt);
				mergeEnd = Math.max(interval.end, mergeEnd);
			}
		}
		ret.add(interPos, new Interval(mergeSt, mergeEnd));
		return ret;
	}
	
	
	
	public List<Interval> merge(List<Interval> intervals) {
        if (intervals==null || intervals.size()<2)
        	return intervals;
        TreeSet<Interval> set = new TreeSet<Interval> (new Comparator<Interval>(){
			public int compare(Interval o1, Interval o2) {
				if (o1.start!= o2.start)
					return o1.start-o2.start;
				return o1.end - o2.end;
			}
        	
        });
        for (Interval interval: intervals){
        	set.add(interval);
        }
        List<Interval> ret = new ArrayList<Interval> ();
        Interval current = set.first();
        for (Interval interval: set){
        	if (interval.start > current.end) {
        		ret.add(current);
        		current = interval; 
        	}
        	else {
        		current.end = Math.max(current.end, interval.end);
        		System.out.println(current);
        	}
        }
        ret.add(current);
        return ret;
    }
	
	public int[] searchRange(int[] nums, int target) {
        return new int[]{binaryLB(nums, target, 0, nums.length-1), 
        		binaryUB(nums, target, 0, nums.length-1)};
    }
	
	public int binaryLB (int [] nums, int target, int lb, int ub) {
		if (target==nums[lb])
			return lb;
		if (lb>=ub)
			return -1;
		else {
			int mid = lb + (ub- lb)/2;
			int left = binaryLB (nums, target, lb, mid);
			if (left!=-1)
				return left;
			else
				return binaryLB(nums, target, mid+1, ub);
		}
	}
	
	public int binaryUB (int [] nums, int target, int lb, int ub) {
		if (target==nums[ub])
			return ub;
		if (lb>=ub)
			return -1;
		else {
			int mid = lb + (ub- lb)/2;
			int right = binaryUB (nums, target, mid+1, ub);
			if (right!=-1)
				return right;
			else
				return binaryUB(nums, target, lb, mid);
		}
	}

	
	public static TreeNode convertArraytoTree(int[] arr, int start, int end){
		if (start > end){
			return null;
		}
		if (start ==end) {
			return new TreeNode(arr[start]);
		}
		else {
			int mid = start + (end-start)/2;
			TreeNode Node = new TreeNode(arr[mid]);
			Node.left = convertArraytoTree(arr, start, mid-1);
			Node.right = convertArraytoTree(arr, mid+1, end);
			return Node;
		}
	}
	
	public static TreeNode insertBST(TreeNode root, int val){
		if (root == null) {
			return new TreeNode(val);
		}
		else if (root.val < val) {
			root.right = insertBST(root.right, val);
		}
		else
			root.left = insertBST(root.left, val);
		return root;
	}
	
	
	public static void inOrderTraverse(TreeNode root){
		if (root == null)
			return;
		inOrderTraverse(root.left);
		System.out.println(root.val);
		inOrderTraverse(root.right);
	}
	
	public List<List<Integer>> pathSum(TreeNode root, int sum) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		if (root==null)
			return result;
		int runningSum = 0;
		Stack<TreeNode> s = new Stack();
		s.push(root);
		while (!s.isEmpty()) {
			root = s.pop();
			runningSum += root.val;
			if (root.right!=null){
				s.push(root.right);
			}
			if (root.left!=null){
				s.push(root.left);
			}
		}
		LinkedList<Integer> currentPath = new LinkedList<Integer>();
	
		search(root, runningSum, sum, currentPath, result);
        return result;
    }
	
	public void search(TreeNode node, int rs, int sum, LinkedList<Integer> currentPath, List<List<Integer>> result) {
		if (node==null) {
			return;
		}
		rs +=node.val;
		currentPath.addLast(node.val);
		if (node.left==null&&node.right==null && rs==sum){
			result.add(copyPath(currentPath));
		}
		
		search(node.left, rs, sum, currentPath, result);
		search(node.right, rs, sum, currentPath, result);
		currentPath.removeLast();
		
	}
	
	
	public ArrayList<Integer> copyPath(LinkedList<Integer> orig){
	    ArrayList<Integer> copy = new ArrayList<Integer>();
	    for (int element:orig){
	        copy.add(element);
	    }
	    return copy;
	}
	
	public int longestIncreasingPath(int[][] matrix) {
		if (matrix==null || matrix.length==0||matrix[0].length==0){
			return 0;
		}
		int rows = matrix.length;
		int cols = matrix[0].length;
		int maxPath = 0;
		int[][] dist = new int[rows][cols];
		for (int i=0; i<rows; i++){
			for (int j=0; j<cols; j++){
				if (dist[i][j]!=0){
					maxPath = Math.max(maxPath, findLongestPath(i, j,  matrix, dist, rows, cols));
				}
			}
		}
		return maxPath;
        
    }
	
	public int findLongestPath(int i, int j, int[][] matrix, int[][] dist, int rows, int cols) {
		if (dist[i][j]!=0){
			return dist[i][j];
		}
		int[] dx = new int[] {-1, 1, 0, 0};
		int[] dy = new int[] {0, 0, 1, -1};
		for (int index = 0; index<4; index++){
			int nx = i + dx[index];
			int ny = j + dy[index];
			if (nx>=0 && nx<rows && ny>=0 && ny < cols &&matrix[nx][ny]>matrix[i][j]) {
				dist[i][j] = Math.max(dist[i][j], findLongestPath(nx, ny, matrix, dist, rows, cols));
			}
		}
		dist[i][j]++;
		return dist[i][j];
		
	}
	
	
	public int numIslands(char[][] grid) {
        if (grid==null||grid.length==0||grid[0].length==0){
        	return 0;
        }
        int rows = grid.length;
        int cols = grid[0].length;
        int count = 0;
        for (int i = 0; i<rows; i++){
        	for (int j = 0; j<cols; j++){
        		if (grid[i][j] == '1'){
        			count++;
        			grid[i][j] = 0;
        			dfs(grid, i, j, rows, cols);
        		}
        	}
        }
        return count;        
    }
	
	public void dfs(char[][] grid, int x, int y, int rows, int cols){
		if (x-1>=0 && grid[x-1][y]=='1'){
			grid[x-1][y] = 0;
			dfs(grid, x-1, y, rows, cols);
		}
		if (x+1<rows && grid[x+1][y]=='1') {
			grid[x+1][y] = 0;
			dfs(grid, x+1, y, rows, cols);
		}
		if (y-1>=0 && grid[x][y-1]=='1') {
			grid[x][y-1] = 0;
			dfs(grid, x, y-1, rows, cols);
		}
		if (y+1<cols && grid[x][y+1]=='1') {
			grid[x][y+1] = 0;
			dfs(grid, x, y+1, rows, cols);
		}
	}
    class Coordination{
    	int x;
    	int y;
    	Coordination(int i, int j){
    		x = i;
    		y = j;
    	}
    }
    
    
    public void flatten(TreeNode root) {
    	if (root==null || (root.left==null&& root.right==null))
    		return;
    	preOrderTraversal(root);
        
    }
    
    private void preOrderTraversal(TreeNode root){
    	if (root==null)
    		return;
    	preOrderTraversal(root.right);
    	rearrange(root);
    	preOrderTraversal(root.left);
    }
    
    private void rearrange(TreeNode root){
    	if (root.left==null)
    		return;
    	TreeNode temp = root.right;
    	root.right = root.left;
    	root.left = null;
    	TreeNode p = root;
    	while (p.right!=null){
    		p = p.right;
    	}
    	p.right = temp;
    }
    
    private TreeNode lastNode = null;

    
    
    int max = 0;
    
    public List<Integer> rightSideView(TreeNode root) {
    	
        List<Integer> rightSide  = new ArrayList<Integer> ();
        if (root==null){
        	return rightSide;
        }
        dfs(root, 1, rightSide);
        return rightSide;
        
    }
    
    public List<List<Integer>> permute(int[] nums) {
    	List<List<Integer>> result = new ArrayList<List<Integer>>();
    	LinkedList<Integer> list = new LinkedList<Integer> ();
    	boolean[] isVisited = new boolean[nums.length];
    	permute(nums, isVisited, result, list);
    	return result;
        
    }
    
    public void permute(int[] nums, boolean[] isVisited, List<List<Integer>> result, LinkedList<Integer> list) {
    	if (list.size()==nums.length){
    		ArrayList<Integer> arr = new ArrayList<Integer>();
    		arr.addAll(list);
    		result.add(arr);
    		return;
    	}
    	for (int i=0; i<isVisited.length; i++){
    		if (!isVisited[i]) {
    			isVisited[i] = true;
    			list.addLast(nums[i]);
    			permute(nums, isVisited, result, list);
    			isVisited[i] = false;
    			list.removeLast();
    		}	
    	}
    }
    
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root==null||p==null||q==null)
        	return null;
        TreeNode lowValueNode = (p.val>q.val)? q: p;
        TreeNode highValueNode = (p.val<=q.val)?q: p;
        if (root.val>=lowValueNode.val && root.val <=highValueNode.val)
        	return root;
        if (root.val>lowValueNode.val && root.val > highValueNode.val)
        	return lowestCommonAncestor(root.left, lowValueNode, highValueNode);
        return lowestCommonAncestor(root.right, lowValueNode, highValueNode);
    }
    
    private void dfs(TreeNode root, int currentLevel, List<Integer> result) {
    	if (root==null)
    		return;
    	if (currentLevel>max){
    		result.add(root.val);
    		max = currentLevel;
    	}
    	dfs(root.right, currentLevel+1, result);
    	dfs(root.left, currentLevel+1, result);
    }
    
    int count_comp;
    int[] root;
    int[] rank;
    public boolean validTree(int n, int[][] edges) {
    	if (n == 1)
    		return true;
        int[] isVisited = new int[n];
        root = new int[n];
        rank = new int[n];
        count_comp = n;
        for (int i = 0; i < n; i++)
        	root[i] = i;
        @SuppressWarnings("unchecked")
		ArrayList<Integer>[] map = (ArrayList<Integer> []) new ArrayList[n];
        for (int[] edge : edges) {
        	if (map[edge[0]] == null) {
        		ArrayList<Integer> v = new ArrayList<>();
        		v.add(edge[1]);
        		map[edge[0]] = v;
        	}
        	else {
        		map[edge[0]].add(edge[1]);
        	}
        	if (map[edge[1]] == null) {
        		ArrayList<Integer> w = new ArrayList<>();
        		w.add(edge[0]);
        		map[edge[1]] = w;
        	}
        	else{
        		map[edge[1]].add(edge[0]);
        	}
        }
        for (int i = 0; i < n; i++) {
        	if (map[i] == null ||(isVisited[i] == 0 && !dfs(map, isVisited, i, -1)))
        		return false;
        }
        return true;
    }
    
    public boolean dfs (ArrayList<Integer>[] map, int[] isVisited, int current, int parent) {
    	//marked it on the path
    	isVisited[current] = 1;
    	for (int neighbor: map[current]) {
    		if (isVisited[neighbor] == 1 && neighbor != parent) 
    			return false;
    		if (isVisited[neighbor] == 0) {
    			union(current, neighbor);
    			if (!dfs(map, isVisited, neighbor, current))
    				return false;
    		}
    	}
    	isVisited[current] = 2;
    	return true;
    }
    
    public boolean union(int i, int j) {
    	while (root[i] != i)  i = root[i];
    	while (root[j] != j)  j = root[j];
    	if (i == j)
    		return false;
    	if (rank[i] < rank[j]) {
    		root[i] = j;
    	}
    	else if (rank[j] < rank [i]) {
    		root[j] = i;
    	}
    	else {
    		root[j] = i;
    		rank[i]++;
    	}
    	return true;
    }
    
    public List<List<Integer>> verticalOrder(TreeNode root) {
    	int minOrder = 0;
        int maxOrder = 0;
    	HashMap<Integer, List<Integer>> map = new HashMap<>();
    	List<List<Integer>> list = new ArrayList<> ();
    	if (root == null)
    		return list;
    	Queue<TreeNode> q = new LinkedList<>();
    	Queue<Integer> q_order = new LinkedList<>();
    	q.offer(root);
    	q_order.offer(0);
    	while (!q.isEmpty()) {
    		int size = q.size();
    		for (int i = 0; i < size; i++) {
    			TreeNode node = q.poll();
    			int level = q_order.poll();
    			if (map.containsKey(level)){
    				map.get(level).add(node.val);
    			}
    			else {
    				ArrayList<Integer> order = new ArrayList<>();
    				order.add(node.val);
    				map.put(level, order);
    			}
    			if (node.left != null) {
    				q.offer(node.left);
    				q_order.offer(level - 1);
    				minOrder = Math.min(level - 1, minOrder);
    			}
    			if (node.right != null) {
    				q.offer(node.right);
    				q_order.offer(level + 1);
    				maxOrder = Math.max(level + 1, maxOrder);
    			}
    		}
    	}
    	for (int i = minOrder; i <= maxOrder; i++) {
    		list.add(map.get(i));
    	}
    	return list;
    }
    
    int count = 0;
    
    public int countUnivalSubtrees(TreeNode root) {
        countUnivalSubtreesHelper(root);
        return count;
    }
    
    public boolean countUnivalSubtreesHelper(TreeNode root) {
        if (root == null)
            return true;
        boolean isLeftUnival = countUnivalSubtreesHelper(root.left);
        boolean isRightUnival = countUnivalSubtreesHelper(root.right);
        if (!isLeftUnival || !isRightUnival)
            return false;
        int left = root.left == null? root.val : root.left.val;
        int right = root.right == null? root.val : root.right.val;
        if (root.val == left && root.val == right) {
            count++;
            return true;
        }
        return false;
    }
    
//    private void preorderTraverse (TreeNode root, int idx,
//    		HashMap<Integer, List<Integer>> map) {
//    	if (root == null) {
//    		return;
//    	}
//    	List<Integer> order = map.containsKey(idx)? map.get(idx):
//    		new ArrayList<>();
//    	order.add(root.val);
//    	minOrder = Math.min(minOrder, idx);
//    	maxOrder = Math.max(idx, maxOrder);
//    	preorderTraverse(root.left, idx - 1, map);
//    	preorderTraverse(root.right, idx + 1, map);
//    }
    
    public int minMeetingRooms(Interval[] intervals) {
        PriorityQueue<Integer> start_p = new PriorityQueue<>();
        PriorityQueue<Integer> end_p = new PriorityQueue<>();
        for (Interval interval: intervals) {
            start_p.offer(interval.start);
            end_p.offer(interval.end);
        }
        int max_overlap = 0;
        int overlap = 0;
        while( !start_p.isEmpty()) {
        	if (start_p.peek() < end_p.peek()){
        		start_p.poll();
        		max_overlap = Math.max(max_overlap, ++overlap);
        	}
        	else {
        		end_p.poll();
        		overlap--;
        	}
        }
        return max_overlap;
    }
    
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> list = new ArrayList<List<Integer>>();
        if (root == null)
        	return list;
        findLeavedHelper(root, list);
        return list;
    }
    
    public int findLeavedHelper(TreeNode root, List<List<Integer>> list) {
    	int leftH = 0;
    	int rightH = 0;
    	if (root.left != null) {
    		leftH = findLeavedHelper(root.left, list);
    	}
    	if (root.right != null) {
    		rightH = findLeavedHelper(root.right, list);
    	}
    	int height = Math.max(leftH,  rightH);
    	if (list.size() <= height) {
    		list.add(new ArrayList<Integer>());
    	}
    	list.get(height).add(root.val);
    	return height + 1;
    }
	
	public static void main(String[] args) {
		Solutions s= new Solutions();
		String[] alien = new String[] {"ze","yf","xd","wd","vd","ua","tt","sz","rd",
                "qd","pz","op","nw","mt","ln","ko","jm","il",
                "ho","gk","fa","ed","dg","ct","bb","ba"};
		System.out.println(s.alienOrder(alien));
	}
		
}


