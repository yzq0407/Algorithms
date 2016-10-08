import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class Solution_100_to_150 {
	
	public ListNode sortList(ListNode head) {
		if (head == null)
			return head;
		ListNode slow = head;
		ListNode fast = head.next;
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		ListNode righthead = slow.next;
		slow.next = null;
		ListNode left = sortList(head);
		ListNode right = sortList(righthead);
		ListNode dummy = new ListNode(0);
		ListNode p = dummy;
		while (left!= null && right != null) {
			if (left.val <= right.val) {
				p.next = left;
				left = left.next;
			}
			else {
				p.next = right;
				right = right.next;
			}
			p = p.next;
		}
		if (left != null) {
			p.next = left;
		}
		else {
			p.next = right;
		}
		return dummy.next;
    }
	
	public int removeDuplicates(int[] nums) {
        if (nums.length <= 2) {
            return nums.length;
        }
        int counter = 1;
        int len = 1;
        int p = 1;
        while (p < nums.length) {
        	if (nums[p] == nums[p - 1]) {
        		counter++;
        	}
        	else {
        		counter = 1;
        	}
        	if (counter <= 2) {
        		nums[len++] = nums[p];
        	}
        	p++;
        }
        System.out.println(Arrays.toString(nums));
        return len;
    }
	
	
	HashMap<Integer, Integer> inorderMap = new HashMap<>();
    HashMap<Integer, Integer> postorderMap = new HashMap<>();
	public TreeNode buildTree(int[] inorder, int[] postorder) {
        if (inorder.length == 0) {
        	return null;
        }
        for (int i = 0; i < inorder.length; i++) {
        	inorderMap.put(inorder[i], i);
        	postorderMap.put(postorder[i], i);
        }
        return buildTreeHelper(inorder, 0, inorder.length - 1, postorder, 
        		0, postorder.length - 1);
    }
	
	public TreeNode buildTreeHelper(int[] inorder, int start1, int end1,
			int[] postorder, int start2, int end2) {
		if (start1 > end1) {
			return null;
		}
		if (start1 == end1) {
			return new TreeNode(inorder[start1]);
		}
		TreeNode root = new TreeNode(postorder[end2]);
		int inorderpos = inorderMap.get(postorder[end2]);
		int leftend = inorderpos - 1;
		int rightstart = inorderpos + 1;
		int leftlength = leftend - start1 + 1;
		int rightlength = end1 - rightstart + 1;
		TreeNode left = buildTreeHelper(inorder, start1, leftend, postorder, start2, 
				start2 + leftlength - 1);
		TreeNode right = buildTreeHelper(inorder, rightstart, end1, postorder, 
				start2 + leftlength, end2 - 1);
		root.left = left;
		root.right = right;
		return root;
				
	}
	
//	public int findCelebrity(int n) {
//        boolean[] celebrities = new boolean[n];
//        int pc = dfs(0, n, celebrities);
//        for (int i = 0; i < n; i++) {
//            if (pc != i && knows(i, pc) && !knows(pc, i))
//                continue;
//            return -1;
//        }
//        return pc;
//    }
//    
//    public int dfs (int i, int n, boolean[] celes) {
//        celes[i] = true;
//        for (int idx = 0; idx < n; idx++) {
//            if (!celes[idx] && knows(i, idx)) {
//                return dfs(idx, n, celes);
//            }
//        }
//        return i;
//    }
	
//	public void connect(TreeLinkNode root) {
//        if (root == null)
//        	return;
//        TreeLinkNode left = root.left;
//        TreeLinkNode right = root.right;
//        TreeLinkNode nephew = findNephew (root);
//        if (left!= null && right != null) {
//        	left.next = right;
//        	right.next = nephew;
//        }
//        else if (right != null) {
//        	right.next = nephew;
//        }
//        else if (left != null) {
//        	left.next = nephew;
//        }
//        
//        connect(right);
//        connect(left);
//    }
//	
//	private TreeLinkNode findNephew (TreeLinkNode root) {
//		while (root.next != null) {
//			root = root.next;
//			if (root.left != null)
//				return root.left;
//			if (root.right != null)
//				return root.right;
//		}
//		return null;
//	}
	
	public ListNode sortList(ListNode head, ListNode end) {
		if (head == end || end == null)
			return head;
		
		ListNode slow = head;
		ListNode fast = head.next;
		while (fast != end && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		ListNode left = sortList(head, slow);
		ListNode right = sortList(slow.next, end);
		ListNode dummy = new ListNode(0);
		ListNode p = dummy;
		while (left != slow.next && right != fast.next) {
			if (left.val <= right.val){
				p.next = left;
				left = left.next;
			}
			else {
				p.next = right;
				right = right.next;
			}
			p = p.next;
		}
		if (left != slow.next) {
			p.next = left;
		}
		if (right != fast.next) {
			p.next = fast;
		}
		return dummy.next;
	}
	
	
	public int findMin(int[] nums) {
		if (nums[0] <= nums[nums.length - 1])
			return nums[0];
		//invariant
		//nums[LB] > nums[UB]
        int LB = 0;
        int UB = nums.length - 1;
        while (UB - LB > 1) {
        	int mid = LB + (UB - LB) / 2;
        	if (nums[mid] < nums[LB]) {
        		LB = mid;
        	}
        	else {
        		UB = mid;
        	}
        }
        return nums[UB];
    }
	
	public List<Integer> getRow(int rowIndex) {
        long current = 1;
        List<Integer> list = new ArrayList<>();
        list.add(1);
        int aux = rowIndex;
        for (int i = 1; i <= rowIndex; i++) {
            current = current * (aux--) / i;
            list.add((int)current);
        }
        return list;
    }
	
	public int numDecodings(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int len = s.length();
        int[] dp = new int[len + 1];
        dp[0] = 1;
        if (s.charAt(0) != '0')
            dp[1] = 1;
        for (int i = 2; i <= len; ++i) {
            int includeOne = 0;
            int includeTwo = 0;
            int two = Integer.parseInt(s.substring(i - 2, i));
            int one = Integer.parseInt(s.substring(i - 1, i));
            if (two <= 26 && two > 0)
                includeTwo = dp[i - 2];
            if (one > 0)
                includeOne = dp[i - 1];
            dp[i] = includeOne + includeTwo;
        } 
        return dp[len];
    }
	
	public int maxProfit(int[] prices) {
		int n = prices.length;
		int[] dp = new int[n];
		int leftMin = prices[0];
		for (int i = 1; i < n; i++) {
			dp[i] = Math.max(prices[i] - leftMin, dp[i - 1]);
			leftMin = Math.min(prices[i], leftMin);
		}
		int rightMax = prices[n - 1];
		int maxProfit = dp[n - 1];
		int rightMaxProf = 0;
		for (int i = n - 2; i >= 1; i--) {
			rightMaxProf = Math.max(rightMaxProf, rightMax - prices[i]);
			maxProfit = Math.max(rightMaxProf, dp[i - 1]);
			rightMax = Math.max(prices[i], rightMax);
		}
		return maxProfit;
    }
	
	public int longestConsecutive(int[] nums) {
		if (nums == null || nums.length == 0)
			return 0;
		UnionFind uf = new UnionFind(nums.length);
		HashMap<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; ++i) {
        	if (map.containsKey(nums[i]))
        		continue;
        	map.put(nums[i], i);
        	if (map.containsKey(nums[i] - 1)) {
        		uf.union(map.get(nums[i] - 1), i);
        	}
        	if (map.containsKey(nums[i] + 1)) {
        		uf.union(map.get(nums[i] + 1), i);
        	}
        }
        return uf.maxCount;
    }
	
	public class UnionFind {
		int[] root;
		int[] depth;
		int[] count;
		int maxCount;
		
		UnionFind (int n) {
			root = new int[n];
			depth = new int[n];
			count = new int[n];
			maxCount = (n == 0)? 0: 1;
			for (int i = 0; i < n; i++) {
				root[i] = i;
				count[i] = 1;
				depth[i] = 1;
			}
		}
		
		void union(int i, int j) {
			while (root[i] != i) {
				i = root[i];
			}
			while (root[j] != j) {
				j = root[j];
			}
			if (i == j) {
				return;
			}
			if (depth[i] <= depth[j]) {
				root[i] = j;
				count[j] += count[i];
				maxCount = Math.max(maxCount, count[j]);
			}
			else if (depth[i] > depth[j]) {
				root[j] = i;
				count[i] += count[j];
				maxCount = Math.max(maxCount, count[i]);
			}
		}
		
		boolean find(int i, int j) {
			while (root[i] != i) {
				i = root[i];
			}
			while (root[j] != j) {
				j = root[j];
			}
			return i == j;
		}
	}
	
	
	
	public boolean isPalindrome(String s) {
		s = s.toLowerCase();
		if (s == null || s.length() < 2)
			return true;
		int i = 0, j = s.length() - 1;
		while (i < j) {
			while (i < s.length() && !Character.isAlphabetic(s.charAt(i)) 
					&& !Character.isDigit(s.charAt(i))) {
				i++;
			}
			while (j >= 0 && !Character.isAlphabetic(s.charAt(j))
					&& !Character.isDigit(s.charAt(j))) {
				j--;
			}
			if (i < s.length() && j >= 0 && s.charAt(i) != s.charAt(j))
				return false;
			i++;
			j--;
		}
		return true;
    }
	
	public List<String> findRepeatedDnaSequences(String s) {
        HashSet<Integer> oneSet = new HashSet<>();
        HashSet<Integer> twoSet = new HashSet<>();
        List<String> rv = new ArrayList<>();
        int[] map = new int[26];
        map['C' - 'A'] = 1;
        map['G' - 'A'] = 2;
        map['T' - 'A'] = 3;
        for (int i = 0; i < s.length() - 9; i++) {
            int v = 0;
            for (int j = i; j < i + 10; j ++) {
                v <<= 2;
                v |= map[s.charAt(j) - 'A'];
            }
            if (!oneSet.add(v) && twoSet.add(v)) {
                rv.add(s.substring(i, i + 10));   
            }
        }
        return rv;
    }
	
	public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        int beforeInd = binarySearch(intervals, newInterval.start);
        int afterInd = binarySearch(intervals, newInterval.end);
        
        System.out.println("before: " + beforeInd + " after: " + afterInd);
        //situations
        // before ---- newInterval ---- beforeEnd ---otherIntervals-- after ---- newInterval ---- afterEnd
        //both they are - 1
        //this case, add the interval into the first
        if (beforeInd == -1 && afterInd == -1) {
            intervals.add(0, newInterval);
            return intervals;
        }
        //before is -1, after is not -1
        if (beforeInd == -1 && afterInd != -1) {
            intervals.subList(0, afterInd).clear();
            Interval interval = intervals.get(0);
            interval.start = newInterval.start;
            interval.end = Math.max(newInterval.end, interval.end);
            return intervals;
        }
        Interval before = intervals.get(beforeInd);
        Interval after = intervals.get(afterInd);
        //both before and after are not -1
        //if before == after, just worry about only one case
        if (beforeInd == afterInd) {
        	if (before.end < newInterval.start) {
        		intervals.add(beforeInd + 1, newInterval);
        		return intervals;
        	}
        	else {
        		before.start = Math.min(before.start, newInterval.start);
                before.end = Math.max(after.end, newInterval.end);
                return intervals;
        	}
        }
        //before < after
        //in this case, the new interval can merge with left, with right, all both
        if (newInterval.start <= before.end && newInterval.start <= after.end) {
        	System.out.println("merge with both");
        	intervals.subList(beforeInd, afterInd).clear();
        	after.start = before.start;
        	after.end = Math.max(newInterval.end, after.end);
        	return intervals;
        }
        //merge with only left
        if (newInterval.start <= before.end) {
        	System.out.println("merge with left");
        	intervals.subList(beforeInd + 1, afterInd + 1).clear();
        	before.end = newInterval.end;
        	return intervals;
        }
        if (newInterval.start <= after.end) {
        	System.out.println("merge with right");
        	intervals.subList(beforeInd + 1, afterInd).clear();
        	after.start = Math.min(newInterval.start, after.start);
        	after.end = Math.max(newInterval.end, after.end);
        	return intervals;
        }
        intervals.subList(beforeInd + 1, afterInd).clear();
        intervals.add(beforeInd + 1, newInterval);
        return intervals;
        
	}
    
    
    //find interval that's has start <= val
    //if all interval has start > val
    //return -1
    public int binarySearch (List<Interval> intervals, int val) {
    	
        int start = 0, end = intervals.size() - 1;
        if (intervals.get(start).start > val)
            return -1;
        //maintain invariant, intervals.start <=  val
        while (end >= start) {
            int mid = start + (end - start) / 2;
            int midVal = intervals.get(mid).start;
            if (midVal <= val) {
                start = mid + 1;
            }
            else {
                end = mid - 1;
            }
        }
        return start - 1;
    }
    
    public List<Interval> insert2(List<Interval> intervals, Interval newInterval) {
        if (intervals.size() == 0 || intervals.get(0).start > newInterval.end) {
            intervals.add(0, newInterval);
            return intervals;
        }
        boolean added = false;
        boolean merged = false;
        List<Interval> rv = new ArrayList<Interval>();
        for (Interval v: intervals) {
            if (v.end < newInterval.start || merged) {
                rv.add(v);
                continue;
            }
            if (!added) {
                newInterval.start = Math.min(newInterval.start, v.start);
                newInterval.end = Math.max(newInterval.end, v.end);
                added = true;
                continue;
            }
            if (added && !merged) {
                if (v.start > newInterval.end) {
                    rv.add(newInterval);
                    rv.add(v);
                    merged = true;
                }
                else if (v.end >= newInterval.end) {
                    merged = true;
                    newInterval.end = v.end;
                    rv.add(newInterval);
                }
                continue;
            }
        }
        if (!merged || !added) {
            rv.add(newInterval);
        }
        return rv;
    }
    
    
    public void reorderList(ListNode head) {
        if (head == null) {
            return;
        }
        ListNode p1 = head;
        ListNode p2 = null;
        ListNode p = head.next;
        p1.next = null;
        int count = 1;
        while (p != null) {
            ListNode temp = p.next;
            p.next = null;
            if (count % 2 == 1) {
                p.next = p2;
                p2 = p;
            }
            else {
                p1.next = p;
                p1 = p;
            }
            p = temp;
            count++;
        }
        p1 = head;
        while (p1 != null && p2 != null) {
            ListNode temp1 = p1.next;
            ListNode temp2 = p2.next;
            p1.next = p2;
            p2.next = temp1;
            p1 = temp1;
            p2 = temp2;
        }
    }
    
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<>();
        LinkedList<UndirectedGraphNode> queue = new LinkedList<>();
        HashSet<UndirectedGraphNode> isVisited = new HashSet<>();
        queue.addLast(node);
        while (queue.size() != 0) {
            UndirectedGraphNode original = queue.removeFirst();
            isVisited.add(original);
            if (!map.containsKey(original)) {
                map.put(original, new UndirectedGraphNode(original.label));
            }
            UndirectedGraphNode clone = map.get(original);
            for (UndirectedGraphNode neighbor: original.neighbors) {
                if (!map.containsKey(neighbor)) {
                    map.put(neighbor, new UndirectedGraphNode(neighbor.label));
                }
                clone.neighbors.add(map.get(neighbor));
                if (!isVisited.contains(neighbor)) {
                    queue.addLast(neighbor);
                }
            }
        }
        return map.get(node);
    }
    
    
    class RandomListNode {
    	int label;
    	RandomListNode next, random;
    	RandomListNode(int x) { this.label = x; }
    };
    public RandomListNode copyRandomList(RandomListNode head) {
        HashMap<RandomListNode, RandomListNode> map = new HashMap<>();
        RandomListNode copyHead = new RandomListNode(head.label);
        map.put(head, copyHead);
        RandomListNode p = head;
        while (p != null) {
            if (p.random != null) {
                if (map.containsKey(p.random)) {
                    copyHead.random = map.get(p.random);
                }
                else {
                    RandomListNode random = new RandomListNode(p.random.label);
                    map.put(p.random, random);
                    copyHead.random = random;
                }
            }
            if (p.next != null) {
                RandomListNode next = map.containsKey(p.next)? map.get(p.next): new RandomListNode(p.next.label);
                copyHead.next = next;
                map.put(p.next, next);
            }
            p = p.next;
            copyHead = copyHead.next;
        }
        return map.get(head);
    }
    
    public int ladderLength(String beginWord, String endWord, Set<String> wordList) {
    	if (beginWord.equals(endWord)) {
    		return 1;
    	}
        LinkedList<String> queue = new LinkedList<>();
        queue.addLast(beginWord);
        int step = 2;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String current = queue.removeFirst();
                char[] array = current.toCharArray();
                for (int j = 0; j < array.length; j++) {
                    char temp = array[j];
                    for (int offset = 0; offset < 26; offset++) {
                        char c = (char)(offset + 'a');
                        if (c == temp){
                            continue;
                        }
                        array[j] = c;
                        String neighbor = String.valueOf(array);
                        //System.out.println(neighbor);
                        if (neighbor.equals(endWord)) {
                        	return step;
                        }
                        if (wordList.contains(neighbor)) {
                            wordList.remove(neighbor);
                            queue.addLast(neighbor);
                        }
                    }
                    array[j] = temp;
                }
            }
            step++;
        }
        return 0;
        
    }
    
    public static class LRUCache {
        private LinkedNode head;
        private LinkedNode tail;
        int capacity;
        int current;
    	HashMap<Integer, LinkedNode> map = new HashMap<>();
    	
        public class LinkedNode {
        	int val;
        	LinkedNode previous;
        	LinkedNode next;
        	int key;
        	
        	public LinkedNode(int val, int key) {
        		this.val = val;
        		this.key = key;
        	}
        }
        
        public LRUCache(int capacity) {
            head = new LinkedNode(0, 0);
            tail = new LinkedNode(0, 0);
            head.next = tail;
            tail.previous = head;
            this.capacity = capacity;
            this.current = 0;
        }
        
        public int get(int key) {
            if (!map.containsKey(key)) {
            	return -1;
            }
            LinkedNode node = map.get(key);
            moveToHead(node);
            return node.val;
        }
        
        public void set(int key, int value) {
            if (map.containsKey(key)) {
            	LinkedNode node = map.get(key);
            	node.val = value;
            	moveToHead(node);
            }
            else {
            	if (capacity == 0) {
            		return;
            	}
            	LinkedNode node = new LinkedNode(value, key);
            	map.put(key, node);
            	moveToHead(node);
            	if (current != capacity) {
            		current++;
            	}
            	else {
            		LinkedNode remove = removeTail();
            		map.remove(remove.key);
            	}
            }
        }
        
        private void moveToHead (LinkedNode node) {
        	if (node.previous != null)
        		node.previous.next = node.next;
        	if (node.next != null)
        		node.next.previous = node.previous;
        	node.previous = head;
        	node.next = head.next;
        	node.previous.next = node;
        	node.next.previous = node;
        }
        
        private LinkedNode removeTail () {
        	LinkedNode toRemove = tail.previous;
        	toRemove.next.previous = toRemove.previous;
        	toRemove.previous.next = tail;
        	return toRemove;
        }
    }
    
    public List<List<String>> findLadders(String beginWord, String endWord, Set<String> wordList) {
    	List<List<String>> rv = new ArrayList<>();
    	Map<String, Set<String>> map = new HashMap<>();
    	HashSet<String> visited = new HashSet<>();
    	wordList.add(endWord);
    	visited.add(beginWord);
    	LinkedList<String> queue = new LinkedList<>();
    	queue.addLast(beginWord);
    	boolean found = false;
    	while (!found) {
    		int size = queue.size();
    		HashSet<String> layer = new HashSet<>();
    		for (int count = 0; count < size; count++) {
    			String node = queue.removeFirst();
    			char[] current = node.toCharArray();
    			map.put(node, new HashSet<>());
    			for (int pos = 0; pos < current.length; pos++) {
    				char temp = current[pos];
    				for (char c = 'a'; c < 'z'; c++) {
    					if (c == temp) continue;
    					current[pos] = c;
    					String neighbor = String.valueOf(current);
    					if (!visited.contains(neighbor) && wordList.contains(neighbor)) {
    						map.get(node).add(neighbor);
    						layer.add(neighbor);
    						queue.addLast(neighbor);
    					}
    				}
    				current[pos] = temp;
    			}
    		}
    		if (layer.contains(endWord)){
    			found = true;
    		}
    		if (queue.isEmpty()){
    			return rv;
    		}
    		visited.addAll(layer);
    	}
    	queue.clear();
    	dfs(rv, map, endWord, queue, beginWord);
    	return rv;
    }
    
    public void dfs(List<List<String>> rv, Map<String, Set<String>> map,
    		String endWord, LinkedList<String> queue, String current){
    	if (current.equals(endWord)){
    		ArrayList<String> copy = new ArrayList<>();
    		copy.addAll(queue);
    		copy.add(endWord);
    		rv.add(copy);
    		return;
    	}
    	queue.add(current);
    	if (map.containsKey(current)) {
    		for (String neighbor: map.get(current)) {
    			dfs(rv, map, endWord, queue, neighbor);
    		}
    	}
    	queue.remove(queue.size() - 1);
    }
    static class Point {
    	int x;
    	int y;
    	Point() { x = 0; y = 0; }
    	Point(int a, int b) { x = a; y = b; }
    }
    
    public int maxPoints(Point[] points) {
        int max = 0;
        Arrays.sort(points, (a, b) -> a.x != b.x? a.x - b.x: a.y - b.y);
        int i = 0;
        while (i < points.length) {
            Point p1 = points[i];
            int z = i + 1;
            while (z < points.length && points[z].x == p1.x && points[z].y == p1.y) z++;
            int base = z - i;
            max = Math.max(max, base);
            HashMap<String, Integer> map = new HashMap<>();
            for (int j = z; j < points.length; j++) {
                Point p2 = points[j];
                Pair slope = getPair(p2.y - p1.y, p2.x - p1.x);
                // p2.y - i / p2.x =  y2 - y1 / x2 - x1   i = y2 - x2 * (y2 - y1)/ (x2 - x1)
                //x2y2 - x1y2 -x2y2 + x2y1
                Pair intercept = getPair(p2.x* p1.y - p1.x * p2.y, p2.x - p1.x);
                String key = slope.toString() + " " + intercept.toString();
                int amount = map.containsKey(key)? map.get(key) + 1: base + 1;
                map.put(key, amount);
                max = Math.max(amount, max);
            }
            i = z;
        }
        return max;
    }
    
    public Pair getPair(int x, int y) {
        if (x == 0 && y == 0)   return new Pair(0, 0);
        if (y == 0)     return new Pair(1, 0);
        if (x == 0)     return new Pair(0, 1);
        int gcd = getGCD(x, y);
        return new Pair(x / gcd, y / gcd);
    }
    
    public int getGCD(int x, int y) {
        if (x < y)  return getGCD(y, x);
        while (x % y != 0) {
            int temp = y;
            y = x % y;
            x = temp;
        }
        return y;
    }
    
    public class Pair {
        int x;
        int y;
        
        public Pair(int i, int j) {
            x = i;
            y = j;
        }
        
        public boolean equals(Object p) {
            return x == ((Pair)p).x && y == ((Pair)p).y;
        }
        
        public int hashCode() {
            return x*31 + y;
        }
        
        public String toString () {
        	return x + "-" + y;
        }
    }
    
    

	public static void main(String[] args) {
		Solution_100_to_150 s = new Solution_100_to_150();
		String[] a = {"hot","dot","dog","lot","log"};
		Set<String> set = new HashSet<>();
		for (String str: a) {
			set.add(str);
		}
		Point[] points = new Point[] {
				new Point(0, 0),
				new Point(1, 1),
				new Point(0, 0),
//				new Point(2, 3),
//				new Point(2, 6),
//				new Point(3, 5),
//				new Point(2, 9),
//				new Point(2, 7)
				
		};
		System.out.println(s.maxPoints(points));
//		while (l1 != null) {
//			System.out.println(l1.val);
//			l1 = l1.next;
//		}
		
	}

}
