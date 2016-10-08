import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;

public class Solution_351_to_400 {
	public boolean isReflected(int[][] points) {
        HashMap<Pair, Integer> map = new HashMap<>();
        int minX = Integer.MAX_VALUE;
        int maxX = Integer.MIN_VALUE;
        for (int[] point: points) {
        	minX = Math.min(minX, point[0]);
        	maxX = Math.max(maxX, point[0]);
        	Pair p = new Pair(point[0], point[1]);
        	int freq = map.containsKey(p)? map.get(p): 0;
        	map.put(p, freq + 1);
        }
        double mid = minX + (maxX - minX) / 2.0;
        for (Pair p : map.keySet()) {
        	int r_x = maxX + minX - p.x;
        	if (!map.containsKey(new Pair(r_x, p.y)))
        		return false;
        }
        return true;
    }
	
	class TrieNode {
	    TrieNode[] children;
	    boolean isWord;
	    // Initialize your data structure here.
	    public TrieNode() {
	        children = new TrieNode[26];
	        isWord = false;
	    }
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
	
	
	public class Logger {
	    
	    HashMap<String, Integer> map;
	    /** Initialize your data structure here. */
	    public Logger() {
	        map = new HashMap<>();
	    }
	    
	    /** Returns true if the message should be printed in the given timestamp, otherwise returns false.
	        If this method returns false, the message will not be printed.
	        The timestamp is in seconds granularity. */
	    public boolean shouldPrintMessage(int timestamp, String message) {
	        if (!map.containsKey(message)) {
	            map.put(message, timestamp);
	            return true;
	        }
	        int last = map.get(message);
	        if (timestamp - last < 10) {
	            return false;
	        }
	        map.put(message, timestamp);
	        return true;
	    }
	}
	
	class Pair {
		int x;
		int y;
		
		Pair(int x, int y) {
			this.x = x;
			this.y = y;
		}
		
		public boolean equals(Object o) {
			Pair other = (Pair) o;
			return x == other.x && y == other.y;
		}
		
		public int hashCode(){
			return 31*x + y;
		}
	}
	
	public int maxEnvelopes(int[][] envelopes) {
        if (envelopes.length == 0) {
            return 0;
        }
        Arrays.sort(envelopes, new Comparator<int[]>(){
           public int compare(int[] a, int[] b) {
               if (a[0] != b[0]) {
                   return a[0] - b[0];
               }
               else {
                   return a[1] - b[1];
               }
           } 
        });
        int n = envelopes.length;
        int[] dp = new int[n];
        int max = 1;
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (envelopes[j][0] <= envelopes[i][0] && envelopes[j][1] <= envelopes[i][1]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            max = Math.max(max, dp[i]);
        }
        return max;
    }
	
	public int countNumbersWithUniqueDigits(int n) {
        if (n < 1) {
            return 0;
        }
        else if (n == 1) {
            return 10;
        }
        if (n > 10) {
            return countNumbersWithUniqueDigits(10);
        }
        int base = 9;
        for (int i = 1; i <n; i++) {
            base *= (9 - i + 1);
        }
        return base + countNumbersWithUniqueDigits(n - 1);
    }
	
	public class HitCounter {
	    TreeMap<Integer, Integer> map;
	    /** Initialize your data structure here. */
	    public HitCounter() {
	        map = new TreeMap<>();
	    }
	    
	    /** Record a hit.
	        @param timestamp - The current timestamp (in seconds granularity). */
	    public void hit(int timestamp) {
	        if (map.containsKey(timestamp)) {
	            map.put(timestamp, map.get(timestamp) + 1);
	        }
	        else {
	            map.put(timestamp, 1);
	        }
	    }
	    
	    /** Return the number of hits in the past 5 minutes.
	        @param timestamp - The current timestamp (in seconds granularity). */
	    public int getHits(int timestamp) {
	        int floor = map.floorKey(timestamp - 299) == null ? 
	        		0 : map.get(map.floorKey(timestamp - 299));
	        int total = map.floorKey(timestamp + 1) == null? 0 :
	        	map.get(map.floorKey(timestamp + 1));
	        return total - floor;
	    }
	}
	
	public int superPow(int a, int[] b) {
        // a % b = x   a * a = (kb + x) * (kb + x) = x^2 % b
        // a % b = x,   c % b = y    ac % b = xy % b
    	int i = 0;
        for (i = 0; i < b.length - 1; i++) {
        	if (b[i] != 0) {
        		break;
        	}
        }
        if (i == b.length - 1 && b[i] == 1) {
        	return a % 1337;
        }
        if (i == b.length - 1 && b[i] == 0) {
        	return 1;
        }
        int copy = a % 1337;
        int[] match = new int[b.length];
        match[match.length - 1] = 1;
        while (compare(b, doubleInt(match)) >= 0) {
        	//System.out.println(Arrays.toString(match));
            copy = (copy * copy) % 1337;
            match = doubleInt(match);
        }
        System.out.println(Arrays.toString(match));
        System.out.println(Arrays.toString(b));
        
        subtract(b, match);
        copy = (copy * (superPow(a, b)) % 1337);
        System.out.println("return "+ copy);
        return copy;
    }
    
    public int compare(int[] a, int[] b) {
    	//System.out.println("comparing " + Arrays.toString(a) + " and " + Arrays.toString(b));
    	if (a.length != b.length) {
    		return a.length - b.length;
    	}
        for (int i = 0; i < a.length; ++i) {
            if (a[i] != b[i]) {
            	//System.out.println("compare result: " + (a[i] - b[i]));
                return a[i] - b[i];
            }
        }
        return 0;
    }
    
    public void subtract(int[] a, int[] b) {
        int borrow = 0;
        for (int i = a.length - 1; i >= 0; --i) {
            a[i] -= borrow;
            if (a[i] < b[i]) {
                borrow = 1;
                a[i] = a[i] + 10 - b[i];
            }
            else {
                borrow = 0;
                a[i] -= b[i];
            }
        }
    }
    
    public int[] doubleInt(int[] a) {
        int carry = 0;
        int[] rv = new int[a.length];
        for (int i = a.length -1; i >= 0; --i) {
            int result = a[i] * 2 + carry;
            carry = result / 10;
            rv[i] = result % 10;
        }
        if (carry != 0) {
        	int[] result = new int[a.length + 1];
        	for (int i = 1; i <= a.length; i++) {
        		result[i] = rv[i - 1];
        	}
        	result[0] = 1;
        	return result;
        }
        return rv;
    }
    
    public boolean isPerfectSquare(int num) {
        if (num < 0) {
            return false;
        }
        if (num < 2) {
            return true;
        }
        long low = 1;
        long high = num / 2;
        //high * high >= num
        while (low <= high) {
            long mid = low + (high - low) / 2;
            long square = mid * mid;
            if (square >= num) {
                high = mid - 1;
            }
            else {
                low = mid + 1;
            }
        }
        high = high + 1;
        //System.out.println(high);
        return high * high == num;
    }
    
    public int getMoneyAmount(int n) {
        int[][] dp = new int[1 + n][1 + n];
        for (int offset = 0; offset < n; offset ++) {
            for (int i = 1; i + offset <= n; i++) {
                int j = i + offset;
                dp[i][j] = Integer.MAX_VALUE;
                if (offset == 0) {
                    dp[i][j] = 0;
                }
                else if (offset == 1) {
                    dp[i][j] = i;
                }
                else if (offset == 2) {
                    dp[i][j] = i + 1;
                }
                else {
                    for (int k = i; k <= j; k++) {
                        int left = k > i? dp[i][k - 1]: 0;
                        int right = k < j? dp[k + 1][j]: 0;
                        dp[i][j] = Math.min(dp[i][j], k + Math.max(left, right));
                    }
                }
            }
        }
        return dp[1][n];
    }
    
    int count = 0;
    public int numberOfPatterns(int m, int n) {
        int[][] board = new int[3][3];
        count = 0;
        int sum = 0;
        dfs(board, m, n, 1, 0, 0);
        sum += (count * 4);
        count = 0;
        dfs(board, m, n, 1, 0, 1);
        sum += (count * 4);
        count = 0;
        dfs(board, m, n, 1, 1, 1);
        sum += count;
        return sum;
    }
    
    public void dfs(int[][] board, int m, int n, int current, int i, int j) {
        if (current >= m) {
            count++;
            if (current == n) {
                return;
            } 
        }
        board[i][j] = 1;
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                if (board[x][y] == 0 && isValid(board, i, j, x, y)){
                    dfs(board, m, n, current + 1, x, y);
                }
            }
        }
        board[i][j] = 0;
    }
    
    public boolean isValid(int[][] board, int x1, int y1, int x2, int y2) {
        int diffX = Math.abs(x1 - x2);
        int diffY = Math.abs(y1 - y2);
        if (diffX <= 1 && diffY <= 1) {
            return true;
        }
        if (diffX == 2 && diffY == 2) {
            return board[1][1] == 1;
        }
        if (diffX == 2 && diffY == 0) {
            return board[1][y1] == 1;
        }
        if (diffY == 2 && diffX == 0) {
            return board[x1][1] == 1;
        }
        return true;
    }
    
    public List<Integer> lexicalOrder(int n) {
        ArrayList<Integer> list= new ArrayList<>();
        preorder(list, 0, n);
        list.remove(0);
        return list;
    }
    
    public void preorder(List<Integer> list, int current, int n) {
    	list.add(current);
    	for (int i = 0; i < 10; i++) {
    		int child = current * 10 + i;
    		if (child != 0 && child <= n) {
    			preorder(list, child, n);
    		}
    	}
    }
    
    public int wiggleMaxLength(int[] nums) {
        int count = 1, i = 1;
        if (nums.length <= 1) {
            return nums.length;
        }
        while (i < nums.length) {
            while (i < nums.length && nums[i] <= nums[i - 1]) i++;
            if (i < nums.length) count++;
            while (i < nums.length && nums[i] >= nums[i - 1]) i++;
            if (i < nums.length) count++;
        }
        int maxCount = count;
        i = 1;
        count = 1;
        while (i < nums.length) {
            while (i < nums.length && nums[i] >= nums[i - 1]) i++;
            if (i < nums.length) count++;
            while (i < nums.length && nums[i] <= nums[i - 1]) i++;
            if (i < nums.length) count++;
        }
        return Math.max(count, maxCount);
    }
    
    public int combinationSum4(int[] nums, int target) {
        if (nums.length == 0)
            return 0;
    	int[] dp = new int[target + 1];
    	Arrays.sort(nums);
    	for (int num: nums) {
    	    if (num <= target)  dp[num] = 1;
    	}
    	for (int i = 1; i <= target; i++) {
    	    for (int num: nums) {
    	        if (num > i)    break;
    	        dp[i] += dp[i - num];
    	    }
    	}
    	return dp[target];
    }
   
    public class SnakeGame {
        
        LinkedList<Coord> snake;
        HashSet<Coord> set;
        private int[][] food;
        private int food_pos;
        int width;
        int height;
        
        /** Initialize your data structure here.
            @param width - screen width
            @param height - screen height 
            @param food - A list of food positions
            E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0]. */
        public SnakeGame(int width, int height, int[][] food) {
            this.food = food;
            this.food_pos = 0;
            this.width = width;
            this.height = height;
            snake = new LinkedList<Coord>();
            Coord first = new Coord(0, 0);
            snake.addLast(first);
            set.add(first);
        }
        
        /** Moves the snake.
            @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down 
            @return The game's score after the move. Return -1 if game over. 
            Game over when snake crosses the screen boundary or bites its body. */
        public int move(String direction) {
            Coord head = snake.peekFirst();
            int x = head.x;
            int y = head.y;
            if (direction.equals("U")) x--;
            if (direction.equals("L")) y--;
            if (direction.equals("R")) y++;
            if (direction.equals("D")) x++;
            if (x < 0 || x >= height || y < 0 || y >= width)    return -1;
            Coord newHead = new Coord(x, y);
            if (set.contains(newHead))  return -1;
            snake.addFirst(newHead);
            set.add(newHead);
            if (x == food[food_pos][0] && y == food[food_pos][1]) {
                food_pos++;
                return food_pos;
            }
            Coord tail = snake.removeLast();
            set.remove(tail);
            return food_pos;
        }
        
        public class Coord {
            int x;
            int y;
            public Coord(int x, int y) {
                this.x = x;
                this.y = y;
            }
            
            public int hashCode() {
                return x*31 + y;
            }
            
            public boolean equals(Object o) {
                return ((Coord)o).x == x && ((Coord)o).y == y;
            }
        }
    }
    
    public int lengthLongestPath(String input) {
    	//input.replace(" ", "\t");
    	//System.out.println(input);
        Stack<Integer> path = new Stack<>();
        int i = 0;
        boolean containsDot = false;
        while (i < input.length() && input.charAt(i) != '\n')  {
            if (input.charAt(i) == '.') containsDot = true;
            i++;
        }
        if (containsDot && i > 0 && input.charAt(i - 1) != '.') return i;
        int current = i;
        path.push(current);
        int max = 0;
        while (i < input.length()) {
            if (input.charAt(i) == '\n') {
                i++;
                int count = 0;
                while (count < path.size() && ((i < input.length() && input.charAt(i) == '\t') || (i < input.length() - 3 && input.substring(i, i + 4).equals("    "))))  {
                    if (input.charAt(i) == '\t')
                    	i++;
                    else {
                    	i+= 4;
                    }
                    count++;
                }
                int j = i;
                boolean isFile = false;
                while (j < input.length() && input.charAt(j) != '\n') {
                    if (input.charAt(j) == '.') isFile = true;
                    j++;
                }
                int fileLength = j - i + 1;
                while (path.size() > count) {
                    current -= path.pop();
                }
                if (path.size() == 0)	fileLength--;
                current += fileLength;
                path.push(fileLength);
                if (isFile && input.charAt(j - 1) != '.')
                    max = Math.max(current, max);
                i = j;
                continue;
            }
            i++;
        }
        return max;
    }
    
    
    public static class RandomizedCollection {
        HashMap<Integer, HashSet<Integer>> map;
        ArrayList<Integer> vector;
        /** Initialize your data structure here. */
        public RandomizedCollection() {
            map = new HashMap<>();
            vector = new ArrayList<>();
        }
        
        /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
        public boolean insert(int val) {
            if (!map.containsKey(val)){
                HashSet<Integer> indices = new HashSet<>();
                indices.add(vector.size());
                map.put(val, indices);
                vector.add(val);
                return true;
            }
            else {
                map.get(val).add(vector.size());
                vector.add(val);
                return false;
            }
        }
        
        /** Removes a value from the collection. Returns true if the collection contained the specified element. */
        public boolean remove(int val) {
            if (!map.containsKey(val))	return false;
            Set<Integer> indices = map.get(val);
            int tail = vector.get(vector.size() - 1);
            int removedIndex = indices.iterator().next();
            if (val != tail) {
            	map.get(tail).remove(vector.size() - 1);
            	map.get(tail).add(removedIndex);
            	vector.set(removedIndex, tail);
            	indices.remove(removedIndex);
            }
            else {
            	indices.remove(vector.size() - 1);
            }
            vector.remove(vector.size() - 1);
            return true;
        }
        
        /** Get a random element from the collection. */
        public int getRandom() {
            Random rn = new Random();
            return vector.get(rn.nextInt(vector.size()));
        }
    }
    
    public int minCut(String s) {
        //if (s.length() < 2) return s.length();
        int n = s.length();
        boolean[][] dp = new boolean[s.length() + 1][s.length() + 1];
        for (int len = 0; len <= n; len++) {
            for (int start = 0; start <= n - len ; start++) {
                // -----start------length -----end
                int end = start + len;
                if (len < 2) {
                    dp[start][end] = true;
                    continue;
                }
                dp[start][end] = dp[start + 1][end - 1] && s.charAt(start) == s.charAt(end - 1);
            }
        }
        if (dp[0][n])   return 0;
        for (int i = 0; i <= n; ++i) {
        	System.out.println(Arrays.toString(dp[i]));
        }
        int[] cuts = new int[s.length() + 1];
        Arrays.fill(cuts, s.length());
        for (int i = 1; i <= n; i++) {
            if (dp[0][i]) {
                cuts[i] = 0;
                continue;
            }
            for (int j = 1; j < i; j++) {
                if (dp[j][i]) {
                    cuts[i] = Math.min(cuts[i], cuts[j] + 1);
                }
            }
        }
        return cuts[n];
    }
    
    
    public boolean isRectangleCover(int[][] rectangles) {
        if (rectangles.length < 1)  return false;
        HashMap<Point, Integer> count = new HashMap<Point, Integer>();
        HashSet<Rect> set = new HashSet<>();
        for (int[] rect: rectangles) {
            Rect r = new Rect(rect[0], rect[1], rect[2], rect[3]);
            if (set.contains(r))    return false;
            set.add(r);
            Point[] corners = new Point[4];
            corners[0] = new Point(rect[0], rect[1]);
            corners[1] = new Point(rect[2], rect[1]);
            corners[2] = new Point(rect[0], rect[3]);
            corners[3] = new Point(rect[2], rect[3]);
            for (Point p: corners) {
                if (count.containsKey(p)) {
                    count.put(p, count.get(p) + 1);
                }
                else    count.put(p, 1);
            }
        }
        //if (count.size() != 4)  return false;
        List<Point> list = new ArrayList<>();
        for (Point p: count.keySet()) {
            int freq = count.get(p);
            if (freq == 1)  list.add(p);
            else if (freq % 2 == 1 || freq > 4) return false;
        }
        if (list.size() != 4)   return false;
        Collections.sort(list);
        //System.out.println(list);
        if (list.get(0).x != list.get(1).x || list.get(0).y != list.get(2).y 
        || list.get(3).x != list.get(2).x || list.get(3).y != list.get(1).y)    return false;
        //System.out.println(list);
        return (list.get(1).y - list.get(0).y == list.get(3).y - list.get(2).y) 
        && (list.get(2).x - list.get(0).x == list.get(3).x - list.get(1).x);
    }
    
    
    public class Point implements Comparable<Point> {
        int x;
        int y;
        
        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
        
        public int hashCode() {
            return 31*x + y;
        }
        
        public boolean equals(Object o) {
            return ((Point)o).x == x && ((Point)o).y == y;
        }
        
        public int compareTo(Point p) {
            return x == p.x? y - p.y: x - p.x;
        }
        
        public String toString() {
        	return "[" + x + ", " + y + "]";
        }
    }
    
    public class Rect {
        int x;
        int y;
        int z;
        int m;
        
        public Rect(int x, int y, int z, int m) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.m = m;
        }
        
        public int hashCode() {
            return 31 * x + 17 * y + 13 * z + m;
        }
        
        public boolean equals(Object o) {
            Rect other = (Rect)o;
            return other.x == x && other.y == y && other.z == z && other.m == m;
        }
    }
	
	public static void main (String[] args) {
		//int[] test = new int[]{1, 2, 3};
		Solution_351_to_400 s = new Solution_351_to_400();
//		for (int i = 0; i < 10000000; ++i) {
//			if (s.isPerfectSquare(i)) {
//				System.out.println(i);
//			}
//		}
		int[][] test = new int[][] {{1,1,2,2},{1,1,2,2},{2,1,3,2}};
		int[][] test2 = new int[][]{{1,1,3,3},{3,1,4,2},{3,2,4,4},{1,3,2,4},{2,3,3,4}};
		int[][] test3 = new int[][] {{0,0,1,1},{0,0,1,1},{0,2,1,3}};
		System.out.println(s.isRectangleCover(test3));
	}
}
