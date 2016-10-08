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
	
	public int findCelebrity(int n) {
        boolean[] isVisited = new boolean[n];
        int current = 0;
        while (true) {
        	isVisited[current] = true;
        	boolean knowOther = false;
        	for (int i = 0; i < n; i++) {
        		if (isVisited[i])
        			continue;
        		if (knows(current, i)) {
        			knowOther = true;
        			break;
        			current = i;
        		}
        	}
        	if (!knowOther) {
        		for (int i = 0; i < n; i++) {
        			if (i!= current && !know(i, current))
        				return -1;
        			if (i!= current && isVisited[i] && knows(current, i))
        				return -1;
        		}
        		return current;
        	}
        }
    }
	
	public int closestValue(TreeNode root, double target) {
		TreeNode node = root.val > target? root.left: root.right;
		if (node == null)
			return root.val;
		int cVal = closestValue(node, target);
		return Math.abs(cVal - target) > Math.abs(root.val - target)? root.val: cVal;
    }
	
	public int closestValue(TreeNode root, double target, int minSoFar){
		if (root == null)
			return minSoFar;
		if (root.val == target)
			return root.val;
		if (Math.abs(root.val - target) <=Math.abs(minSoFar - target)){
			minSoFar = root.val;
		}
		if (root.val < target) {
			return closestValue(root.right, target, minSoFar);
		}
		else {
			return closestValue(root.left, target, minSoFar);
		}
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
	
	
	public static void main(String[] args) {
		Solutions s= new Solutions();
		String[] test = new String[] {"abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"};

		System.out.println(s.groupStrings(test));
	}
		
}


