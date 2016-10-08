import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.awt.Rectangle;
import java.util.*;

import javax.swing.text.html.HTMLDocument.Iterator;

public class Solution_201_to_250 {
//	class TrieNode {
//		TrieNode[] children = new TrieNode[26];
//		String s = null;
//	}
//	
//	public void put(TrieNode root, String s) {
//	    if (s != null && s.length() > 0)
//		    put(root, s, -1);
//	}
	
	class TrieNode {
	    TrieNode[] children;
	    boolean isWord;
	    // Initialize your data structure here.
	    public TrieNode() {
	        children = new TrieNode[26];
	        isWord = false;
	    }
	}
	
	public List<Integer> diffWaysToCompute(String input) {
		ArrayList<Integer> numList = new ArrayList<>();
		ArrayList<Character> opList = new ArrayList<>();
		int i = 0;
		while (i < input.length()) {
			int j = i + 1;
			while (j < input.length() && Character.isDigit(input.charAt(j))){
				j++;
			}
			numList.add(Integer.parseInt(input.substring(i, j)));
			if (j < input.length()) {
				opList.add(input.charAt(j));
			}
			i = j + 1;
		}
		Integer[] nums = numList.toArray(new Integer[numList.size()]);
		Character[] ops = opList.toArray(new Character[opList.size()]);
		return diffWaysToCompute(nums, ops, 0, nums.length - 1);
    }
	
	public List<Integer> diffWaysToCompute(Integer[] nums, Character[] operators, int from, int to) {
		List<Integer> result = new ArrayList<>();
		if (from == to) {
			result.add(nums[from]);
			return result;
		}
		if (from == to - 1) {
			result.add(compute(nums[from], nums[to], operators[from]));
			return result;
		}
		for (int bp = from; bp < to; bp++) {
			List<Integer> left = diffWaysToCompute(nums, operators, from, bp);
			List<Integer> right = diffWaysToCompute(nums, operators, bp + 1, to);
			for (int ln: left) {
				for (int rn: right) {
					result.add(compute(ln, rn, operators[bp]));
				}
			}
		}
		return result;
	}
	
	public int compute(int left, int right, char operator) {
		if (operator == '+') {
			return left + right;
		}
		else if (operator == '-') {
			return left - right;
		}
		else {
			return left * right;
		}
	}
	
	public String shortestPalindrome(String s) {
		if (s.length() == 0) {
			return "";
		}
		char[] array = s.toCharArray();
		int i = 0;
		int j = s.length() -1;
		while (i < j) {
			char temp = array[i];
			array[i++] = array[j];
			array[j--] = temp;
		}
		String rev = String.valueOf(array);
		String comb = s + '#' + rev;
		int[] next = new int[comb.length()];
		next[0] = 0;
		for (i = 1; i < next.length; ++i){
			j = next[i - 1];
			while (j > 0 && comb.charAt(j) != comb.charAt(i)) {
				j = next[j - 1];
			}
			next[i] = (comb.charAt(j) == comb.charAt(i))? j + 1: 0;
		}
		System.out.println(Arrays.toString(next));
		return rev.substring(0, rev.length() - next[next.length - 1]) + s;
    }
	
	public boolean isHappy(int n) {
        HashSet<Integer> set = new HashSet<>();
        set.add(n);
        while (n != 1) {
            int sum = 0;
            System.out.println(n);
            while (n != 0) {
                int digit = n % 10;
                sum += (digit * digit);
                n /= 10;
            }
            if (set.contains(sum)){
                return false;
            }
            set.add(sum);
            n = sum;
        }
        return true;
    }

	public class Trie {
	    private TrieNode root;

	    public Trie() {
	        root = new TrieNode();
	    }

	    // Inserts a word into the trie.
	    public void insert(String word) {
	        if (word.length() == 0){
	            root.isWord = true;
	        }
	        root.children[word.charAt(0) - 'a'] = insert(root.children[word.charAt(0) - 'a'], word, 0);
	    }
	    
	    private TrieNode insert(TrieNode node, String word, int pos) {
	        if (node == null) {
	            node = new TrieNode();
	        }
	        if (pos == word.length() - 1){
	            node.isWord = true;
	        }
	        else {
	            node.children[word.charAt(pos + 1) - 'a'] = insert(node.children[word.charAt(pos + 1) - 'a'], word, pos + 1);
	        }
	        return node;
	    }

	    // Returns if the word is in the trie.
	    public boolean search(String word) {
	        if (word.length() == 0) {
	            return root.isWord;
	        }
	        return search(word, 0, root.children[word.charAt(0) - 'a']);
	    }
	    
	    private boolean search(String word, int pos, TrieNode node) {
	        if (node == null) {
	            return false;
	        }
	        if (pos == word.length() - 1) {
	            return node.isWord;
	        }
	        return search(word, pos + 1, node.children[word.charAt(pos + 1) - 'a']);
	    }

	    // Returns if there is any word in the trie
	    // that starts with the given prefix.
	    public boolean startsWith(String prefix) {
	        if (prefix.length() == 0) {
	            return true;
	        }
	        return startsWith(prefix, 0, root.children[prefix.charAt(0) - 'a']);
	    }
	    
	    private boolean startsWith(String word, int pos, TrieNode node) {
	        if (node == null) {
	            return false;
	        }
	        if (pos == word.length() - 1) {
	            return true;
	        }
	        return startsWith(word, pos + 1, node.children[word.charAt(pos + 1) - 'a']);
	    }
	}
	
//	int count = 0;
//	public int strobogrammaticInRange(String low, String high) {
//		if (low.equals("0"))
//			count++;
//        expand(low, high, "8");
//        expand(low, high, "1");
//        expand(low, high, "0");
//        expand(low, high, "");
//        return count;
//    }
//	
//	public void expand(String low, String high, StringBuilder currentSB) {
//		//System.out.println(current);
//		String current = currentSB.toString();
//		if (numericalComp(current, high) > 0) {
//			return;
//		}
//		if (numericalComp(current, low) >= 0 && numericalComp(current, high) <= 0 
//				&& current.charAt(0) != '0') {
//			count++;
//		}
//		expand(low, high, currentSB.a);
//		expand(low, high, "1" + current + "1");
//		expand(low, high, "0" + current + "0");
//		expand(low, high, "9" + current + "6");
//		expand(low, high, "6" + current + "9");
//		
//	}
	
	public int numericalComp (String a, String b) {
		if (a.length() != b.length()) {
			return a.length() - b.length();
		}
		return a.compareTo(b);
	}
	
	public boolean containsNearbyDuplicate(int[] nums, int k) {
        int len = nums.length;
        Integer[] idxes = new Integer[len];
        for (int i = 0; i < len; i++) {
        	idxes[i] = i;
        }
        Arrays.sort(idxes, new Comparator<Integer>(){

			@Override
			public int compare(Integer o1, Integer o2) {
				if (nums[o1] != nums[o2])
					return nums[o1] - nums[o2];
				// TODO Auto-generated method stub
				return o1 - o2;
			}
        	
        });
        for (int i = 1; i < len; ++i) {
        	if (nums[idxes[i]] != nums[idxes[i - 1]])
        		continue;
        	if (idxes[i] - idxes[i - 1] <= k)
        		return true;
        }
        return false;
    }
	
//	private TrieNode put (TrieNode root, String s, int pos) {
//		if (root == null)
//			root = new TrieNode();
//		if (pos == s.length() - 1) {
//			root.s = s;
//		}
//		else {
//			root.children[s.charAt(pos + 1) - 'a'] = put(root.children[s.charAt(pos + 1) - 'a'],
//					s, pos + 1);
//		}
//		return root;
//	}
	
//	public List<String> findWords(char[][] board, String[] words) {
//		TrieNode root = new TrieNode();
//		for (String word: words) {
//			put(root, word);
//		}
//		List<String> list = new ArrayList<> ();
//		if (board == null || board.length == 0)
//			return list;
//		for (int i = 0; i < board.length; i++) {
//			for (int j = 0; j < board[0].length; j++) {
//				dfs(list, root.children[board[i][j] - 'a'], board, i, j);
//			}
//		}
//		return list;
//    }
	
	
//	public void dfs(List<String> list, TrieNode node, char[][] board, int i, int j) {
//		if (node == null)
//			return;
//		if (node.s != null) {
//			list.add(node.s);
//			node.s = null;
//		}
//		char temp = board[i][j];
//		board[i][j] = '#';
//		for (int d = 0; d < 4; d++) {
//			int nx = i + dx[d];
//			int ny = j + dy[d];
//			if (nx >=0 && nx < board.length && ny >= 0 && ny < board[0].length
//					&& board[nx][ny] != '#'){
//				dfs(list, node.children[board[nx][ny] - 'a'], board, nx, ny);
//			}
//		}
//		board[i][j] = temp;
//	}
	
	public boolean isIsomorphic(String s, String t) {
		if (s == null && t == null)
			return true;
		if (s.length() != t.length())
			return false;
		HashMap<Character, Character> s_to_t = new HashMap<>();
		HashMap<Character, Character> t_to_s = new HashMap<>();
		for (int idx = 0; idx < s.length(); ++idx) {
			char charS = s.charAt(idx);
			char charT = t.charAt(idx);
			if (s_to_t.containsKey(charS) && t_to_s.containsKey(charT)) {
				if (s_to_t.get(charS) != charT || t_to_s.get(charT) != charS)
					return false;
			}
			else if (!s_to_t.containsKey(charS)&& !t_to_s.containsKey(charT)) {
				s_to_t.put(charS, charT);
				t_to_s.put(charT, charS);
			}
			else
				return false;
		}
		return true;
    }
	
	
	public static class WordDictionary {
		TrieNode2 root;
		
		public WordDictionary (){
			root = new TrieNode2();
		}

	    // Adds a word into the data structure.
	    public void addWord(String word) {
	        addWord(root, word, 0);
	    }
	    
	    public void addWord(TrieNode2 node, String word, int pos) {
	    	if (pos == word.length()) {
	    		node.isTerminated = true;
	    		return;
	    	}
	    	if (node.children[word.charAt(pos) - 'a'] == null) {
	    		node.children[word.charAt(pos) - 'a'] = new TrieNode2();
	    	}
	    	addWord(node.children[word.charAt(pos) - 'a'], word, pos + 1);
	    }

	    // Returns if the word is in the data structure. A word could
	    // contain the dot character '.' to represent any one letter.
	    public boolean search(String word) {
	        return search(word, root, 0);
	    }
	    
	    private boolean search(String word, TrieNode2 node, int pos) {
    		if (node == null)
	    		return false;
	    	if (pos == word.length()) {
	    		return node.isTerminated;
	    	}
	    	if (word.charAt(pos) != '.') {
	    		return search(word, node.children[word.charAt(pos++) - 'a'], pos);
	    	}
	    	else {
	    		for (TrieNode2 child: node.children) {
	    			if (child!= null && search(word, child, pos + 1))
	    				return true;
	    		}
	    		return false;
	    	}
	    }
	}
	
	public static class TrieNode2 {
		char val;
		boolean isTerminated = false;
		TrieNode2[] children = new TrieNode2[26];
		
	}
	
	public int rangeBitwiseAnd(int m, int n) {
        if (m == 0 || n == 0)
            return 0;
        int result = 0;
        for (int i = 31; i >= 0; i--) {
            int cover = 1 << i;
            
            int digit1 = ((cover & m) == 0)? 0: 1;
            int digit2 = ((cover & n) == 0)? 0: 1;
            System.out.println("d1: " + digit1 + "  d2: "+ digit2);
            if (digit1 != digit2) {
                return result << (i + 1);
            }
            else {
                result = (result << 1) + (digit1 & digit2);
            }
            //System.out.println(result);
        }
        return result;
    }
	
	int minX;
    int maxX;
    int minY;
    int maxY;
    int[] dx = {-1, 1, 0, 0};
	int[] dy = {0, 0, -1, 1};
	public int minArea(char[][] image, int x, int y) {
        if (image.length == 0) {
            return 0;
        }
        minX = maxX = x;
        minY = maxY = y;
        dfs(image, x, y);
        return (maxX - minX + 1) * (maxY - minY + 1);
    }
	
	public void dfs(char[][] image, int x, int y){
		image[x][y] = '0';
		minX = Math.min(x, minX);
		minY = Math.min(y, minY);
		maxX = Math.max(x, maxX);
		maxY = Math.max(y, maxY);
		for (int d = 0; d < 4; ++d) {
			int nx = dx[d] + x;
			int ny = dy[d] + y;
			if (nx < image.length && nx >= 0
					&& ny < image[0].length && ny >=0 && image[nx][ny] == '1'){
				dfs(image, nx, ny);
			}
		}
	}
	
	public class WordDistance {
	    
	    HashMap<String, List<Integer>> map;
	    public WordDistance(String[] words) {
	        for (int i = 0; i < words.length; i++) {
	            String word = words[i];
	            if (map.containsKey(word)){
	                map.get(word).add(i);
	            }
	            else {
	                List<Integer> list = new ArrayList<>();
	                list.add(i);
	                map.put(word, list);
	            }
	        }
	    }
	    

	}
	
	public int[] maxSlidingWindow(int[] nums, int k) {
        //one 
        LinkedList<Integer> ls = new LinkedList<>();
        for (int i = 0; i < k; i++) {
            while (!ls.isEmpty() && nums[ls.peekFirst()] <= nums[i]){
            	ls.removeFirst();
            }
            ls.addFirst(i);
        }
        int[] rv = new int[nums.length - k + 1];
        for (int i = 0; i < rv.length; i++) {
        	//nums value we are looking at is i + k 
        	//first we want to find the largest
        	int largest = ls.peekLast();
        	if (largest == i) {
        		ls.removeLast();
        	}
        	rv[i] = nums[largest];
        	if (i + k < nums.length) {
        		int num = nums[i + k];
        		while (!ls.isEmpty() && nums[ls.peekFirst()] <= num) {
        			ls.removeFirst();
        		}
        		ls.addFirst(i + k);
        	}
        }
        return rv;
    }
	
	public int[] productExceptSelf(int[] nums) {
        if (nums.length == 0) {
            return nums;
        }
        int[] rv = new int[nums.length];
        recursive(nums, 0, 1, rv);
        return rv;
    }
    
    public int recursive(int[] nums, int pos, int prev, int[] result) {
        if (pos == nums.length - 1) {
            result[pos] = prev;
            return nums[pos];
        }
        int follow = recursive(nums, pos + 1, prev * nums[pos], result);
        result[pos] = prev * follow;
        return follow * nums[pos];
    }
    
//    public int calculate(String s) {
//        Stack<Integer> signs = new Stack<> ();
//        int sign = 1;
//        int last_sign = 1;
//        int i = 0;
//        int sum = 0;
//        PriorityQueue<Integer> lowerBracket = new PriorityQueue<>(Collections.reverseOrder());
//        while (i < s.length()) {
//            if (s.charAt(i) == ' ') i++;
//            else if (s.charAt(i) == '(') {
//                signs.push(sign);
//                sign *= last_sign;
//                last_sign = 1;
//                i++;
//            }
//            else if (s.charAt(i) == ')') {
//                sign = signs.pop();
//                i++;
//            }
//            else if (s.charAt(i) == '+') {
//                last_sign = 1;
//                i++;
//            }
//            else if (s.charAt(i) == '-') {
//                last_sign = -1;
//                i++;
//            }
//            else {
//                int j = i + 1;
//                while (j < s.length() && Character.isDigit(s.charAt(j))) {
//                    j++;
//                }
//                int number = Integer.parseInt(s.substring(i, j));
//                sum += (last_sign * sign * number);
//                i = j;
//                //System.out.println("number is " + number + " last sign: "+ last_sign + " sign: " + sign);
//            }
//        }
//        return sum;
//    }
    
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (nums.length < 2 || k < 1 || t < 0) {
            return false;
        }
        HashMap<Long, Long> map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            long remapped = (long)nums[i] - (long)Integer.MIN_VALUE;
            long bucket = remapped / ((long)t + 1);
            //System.out.println(bucket);
            if (map.containsKey(bucket) || 
            (map.containsKey(bucket - 1) && remapped - map.get(bucket - 1) <= t) ||
            (map.containsKey(bucket + 1) && map.get(bucket + 1) - remapped <= t)) {
            	System.out.println(map);
                return true;
            }
            if (map.entrySet().size() >= k) {
                map.remove(((long)nums[i - k] - (long)Integer.MIN_VALUE) / ((long)t + 1));
            }
            map.put(bucket, remapped);
        }
        return false;
    }
    
    public List<int[]> getSkyline(int[][] buildings) {
        List<int[]> rv = new ArrayList<>();
        TreeSet<Rectangle> map = new TreeSet<>();
        Point[] points = new Point[buildings.length * 2];
        for (int i = 0; i < buildings.length; ++i) {
            points[2 * i] = new Point(buildings[i][0], 1, i, buildings[i][2]);
            points[2 * i + 1] = new Point(buildings[i][1], 2, i, buildings[i][2]);
        }
        Arrays.sort(points);
        for(Point point: points) {
            //entry
            if (point.io == 1) {
                //add into the treemap
                if (map.isEmpty() || map.last().height < buildings[point.idx][2]) {
                	rv.add(new int[]{point.x, buildings[point.idx][2]});
                }
                map.add(new Rectangle(buildings[point.idx][2], point.idx));
            }
            else {
            	map.remove(new Rectangle(buildings[point.idx][2], point.idx));
            	if (!map.isEmpty() && map.last().height < buildings[point.idx][2]){
            		rv.add(new int[]{point.x, map.last().height});
            	}
            	if (map.isEmpty()) {
            		rv.add(new int[]{point.x, 0});
            	}
            }
        }
        return rv;
        
    }
    
    public class Point implements Comparable<Point> {
        int x;
        int height;
        int io;
        int idx;
        
        public Point (int x, int io, int idx, int height) {
            this.x = x;
            this.io = io;
            this.idx = idx;
            this.height = height; 
        }
        
        public int compareTo(Point p) {
            if (x != p.x) {
                return x - p.x;
            }
            if (io != p.io)
            	return io - p.io;
            if (io == 1) {
            	return p.height - height;
            }
            return height - p.height;
        }
    }
    
    public class Rectangle implements Comparable<Rectangle> {
        int height;
        int idx;
        
        
        public Rectangle(int height, int idx) {
            this.height = height;
            this.idx = idx;
        }
        
        public int compareTo (Rectangle rect) {
            if (height != rect.height) {
                return height - rect.height;
            }
            return idx - rect.idx;
        }
        
        public boolean equalTo(Rectangle rect) {
            return height == rect.height && idx == rect.idx;
        }
        
    }
    
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }
        if (left == null && right == null) {
            return null;
        }
        return left == null? right: left;
    }
    
    public int calculate(String s) {
    	s = "+" + s.replace(" ", "");
    	int lastValue = 0;
    	int total = 0;
    	int pos = 0;
    	while (pos < s.length()) {
    		char operator = s.charAt(pos++);
    		int j = pos + 1;
    		while (j < s.length() && Character.isDigit(s.charAt(j))) j++;
    		int number = Integer.parseInt(s.substring(pos, j));
    		if (operator == '+') {
    			lastValue = number;
    			total += number;
    		}
    		if (operator == '-') {
    			lastValue = -number;
    			total -= number;
    		}
    		if (operator == '*') {
    			total -= lastValue;
    			lastValue *= number;
    			total += lastValue;
    		}
    		if (operator == '/') {
    			total -= lastValue;
    			lastValue /= number;
    			total += lastValue;
    		}
    		pos = j;
    	}
    	return total;
    }
    
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        TreeNode node = root;
        int level = 0;
        while (node != null) {
            node = node.left;
            level++;
        }
        return search(root, level - 1);
        
    }
    
    public int search (TreeNode root, int level) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        TreeNode p = root;
        p = p.left;
        for (int i = 0; i < level - 1; i++) {
            p = p.right;
        }
        if (p == null) {
        	System.out.println(level);
            return (1 << (level - 1)) + search(root.left, level - 1);
        }
        else {
        	System.out.println(level);
            return (1 << level) + search(root.right, level - 1);
        }
    }
    
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> rv = new ArrayList<>();
        if (k > 9 || n > 45 || k <= 0)    return rv;
        ArrayList<Integer> list = new ArrayList<>();
        dfs(k, n, rv, list, 9);
        return rv;
    }
    
    public void dfs(int k, int current, List<List<Integer>> rv, ArrayList<Integer> list, int bound) {
        //if (current > (2* bound - k + 1) * k / 2)   return;
        if ((k == 0 && current != 0 )|| (current == 0 && k != 0)) return;
        if (k == 0 && current == 0) {
            ArrayList<Integer> copy = new ArrayList<>();
            copy.addAll(list);
            rv.add(copy);
            return;
        }
        for (int factor = Math.min(bound, current); factor >= 1; factor--) {
            list.add(factor);
            dfs(k - 1, current - factor, rv, list, factor - 1);
            list.remove(list.size() - 1);
        }
    }
    
    
    public void wiggleSort(int[] nums) {
    	int median = quickSelection(nums, 0, nums.length - 1, nums.length / 2);
    	System.out.println(Arrays.toString(nums));
    	int left = 0;
    	int right = nums.length - 1;
    	int i = 0;
    	
    	while (i <= right) {
    		//System.out.println("index: " + i + " mapped to: " + transfer(i, nums));
    		//System.out.println(Arrays.toString(nums));
    		if (nums[transfer(i, nums)] > median) {
    			swap(nums, transfer(left++, nums), transfer(i++, nums));
    		}
    		else if (nums[transfer(i, nums)] < median) {
    			swap(nums, transfer(right--, nums), transfer(i, nums));
    		}
    		else {
    			i++;
    		}
    	}
    	System.out.println(Arrays.toString(nums));
    }
    
    public int transfer(int idx, int[] nums) {
    	return (2 * idx + 1) % (nums.length | 1);
    }
	
    public int findKthLargest(int[] nums, int k) {
        return quickSelection(nums, 0, nums.length - 1, k);
    }
    
    public int quickSelection(int[] nums, int lb, int ub, int k) {
    	//System.out.println(Arrays.toString(nums));
    	//System.out.println("pivot :   " + nums[lb]);
        if (ub == lb && lb == k)   return nums[lb];
        if (ub <= lb)	return -1;
        int pivot = nums[lb];
        int i = lb + 1;
        int right = ub;
        int left = lb;
        while (left < right) {
            if (nums[i] <= pivot) {
                nums[left++] = nums[i++];
            }
            else {
                swap(nums, right--, i);
            }
        }
        nums[left] = pivot;
        if (left == k)  return pivot;
        else if (left > k)   return quickSelection(nums, lb, left - 1, k);
        else   return quickSelection(nums, left + 1, ub, k);
    }
    
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    
	public static void main(String[] args) {
		Solution_201_to_250 s = new Solution_201_to_250();
		int[][] test = new int[][] {{1, 2, 1}, {1, 2, 2}, {1, 2, 3}};
//		for (int[] point: s.getSkyline(test)) {
//			System.out.println(point[0] + "  " + point[1]);
//		}
		TreeNode root = new TreeNode(1);
		TreeNode root2 = new TreeNode(2);
		TreeNode root3 = new TreeNode(3);
		TreeNode root4 = new TreeNode(4);
		root.left = root2;
		root.right = root3;
		root2.left = root4;
		s.wiggleSort(new int[]{1, 3, 2, 2, 3, 1});
		//System.out.println(s.wiggleSort(new int[]{1, 5, 1, 1, 6, 4}));
	}
	

}
