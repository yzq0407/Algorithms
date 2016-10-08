import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

public class Solution_251_to_300 {
	
	public class Vector2D implements Iterator<Integer> {
		Iterator<List<Integer>> outer_iter;
		Iterator<Integer> inner_iter;

	    public Vector2D(List<List<Integer>> vec2d) {
	        outer_iter = vec2d.iterator();
	        while (outer_iter.hasNext() && (inner_iter == null || !inner_iter.hasNext())) {
	        	inner_iter = outer_iter.next().iterator();
	        }
	    }

	    @Override
	    public Integer next() {
	        Integer ret = inner_iter.next();
	        while (outer_iter.hasNext() && (inner_iter == null || !inner_iter.hasNext())) {
	        	inner_iter = outer_iter.next().iterator();
	        }
	        return ret;
	        
	    }

	    @Override
	    public boolean hasNext() {
	        return inner_iter != null && inner_iter.hasNext();
	    }
	}
	
	public List<String> generatePalindromes(String s) {
		List<String> rv = new ArrayList<> ();
		int[] count = new int[128];
		for (int i = 0; i < s.length(); i++) {
			count[s.charAt(i)]++;
		}
		String core = "";
		int countOdd = 0;
		StringBuilder sb = new StringBuilder();
		for (int idx = 0; idx < 128; idx++) {
			if (count[idx] % 2 != 0) {
				core = String.valueOf((char) idx);
				if (++countOdd > 1)
					return rv;
			}
			count[idx] /= 2;
			for (int i = 0; i < count[idx]; ++i) {
				sb.append((char) idx);
			}
		}
		char[] array = sb.toString().toCharArray();
		List<String> perms = new ArrayList<>();
		getPermutation(array, 0, perms);
		for (String str: perms) {
			rv.add(str + core + reverseString(str));
		}
		return rv;
		
    }
	
	public void getPermutation(char[] array, int i, List<String> list) {
		if (array.length == 0) {
	        list.add("");
	        return;
	    }
		if (i == array.length - 1) {
			list.add(String.valueOf(array));
			return;
		}
		for (int idx = i; idx < array.length; idx++) {
			swapIdx(array, i, idx);
			if (i == idx || array[i] != array[idx])
				getPermutation(array, i + 1, list);
			swapIdx(array, idx, i);
		}
	}
	
	public boolean canAttendMeetings(Interval[] intervals) {
        Arrays.sort(intervals, new Comparator<Interval>(){

			@Override
			public int compare(Interval o1, Interval o2) {
				// TODO Auto-generated method stub
				return o1.start - o2.start;
			}
        	
        });
        for (int i = 1; i < intervals.length; ++i) {
        	if (intervals[i].start < intervals[i - 1].end) {
        		return false;
        	}
        }
        return true;
    }
	
	
	public int hIndex(int[] citations) {
        if (citations.length == 0)
            return 0;
        int low = 0;
        int high = citations.length - 1;
        //the h-index can only be 0 --- citaions.length
        //if the h-index is h, what we have is there are h citations greater than h
        //citations[length - h] >= h;
        //also the rest is less then equal to h
        //citaions[length - h - 1] <= h
        //apparently we want to find the largest one, we keep citations[length - low] >= low
        //qualified citations[h] == length - h
        //low is paper number >= h_low
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int numCitations = citations.length - mid;
            if (citations[numCitations] >= mid) {
                low = mid + 1;
            }
            else {
                high = mid - 1;
            }
        }
        return low - 1;
    }
	
	
	
	public void swapIdx(char[] array, int i, int j) {
		char temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
	
	public String reverseString(String s) {
		StringBuilder sb = new StringBuilder();
		for (int i = s.length() - 1; i >= 0; --i) {
			sb.append(s.charAt(i));
		}
		return sb.toString();
	}
	
//	public int hIndex(int[] citations) {
//        if (citations == null || citations.length == 0)
//            return 0;
//        Arrays.sort(citations);
//        for (int offset = 1; offset < citations.length; offset++) {
//            //there are offset numbers of citations with h >= citations[citations.length - offset]
//            //there are citations.length - offset + 1 numbers of citations with h <= citations[citations.length -offset]
//            //this means if offset is h index
//            //offset <= citations[citations.length - offset]
//            //citations.length - offset 
//        	if (citations[citations.length - offset] >= offset 
//        	    && citations[citations.length - offset - 1] <= offset)
//        		return offset;
//        }
//        return Math.min(citations.length, citations[0]);
//    }
	
	public int minCost(int[][] costs) {
		int dp[] = new int[3];
		int aux[] = new int[3];
		for (int i = 0; i < costs.length; i++) {
			aux[0] = costs[i][0] + Math.min(dp[1], dp[2]);
			aux[1] = costs[i][1] + Math.min(dp[0], dp[2]);
			aux[2] = costs[i][2] + Math.min(dp[0], dp[1]);
			dp = aux;
			aux = new int[3];
		}
		return Math.min(Math.min(dp[0], dp[1]), dp[2]);
    }
	
	int order = 0;
	public String alienOrder(String[] words) {
		if (words.length == 0) {
			return "";
		}
		if (words.length == 1) {
			return words[0];
		}
		HashMap<Character, Set<Character>> map = new HashMap<>();
		for (int i = 0; i < words.length - 1; i++) {
			String current = words[i];
			String next = words[i + 1];
			int j = 0;
			for (j = 0; j < Math.min(current.length(), next.length()); ++j) {
				char c = current.charAt(j);
				char n = next.charAt(j);
				if (!map.containsKey(c))
					map.put(c, new HashSet<Character>());
				if (!map.containsKey(n))
					map.put(n, new HashSet<Character>());
				if (c != n) {
					map.get(c).add(n);
					break;
				}
			}
			while (j < current.length()) {
			    if (!map.containsKey(current.charAt(j))){
			        map.put(current.charAt(j), new HashSet<Character>());
			    }
			    j++;
			}
		}
		String last = words[words.length - 1];
		for (int j = 0; j < last.length(); j++){ 
		    if (!map.containsKey(last.charAt(j))){
		        map.put(last.charAt(j), new HashSet<Character>());
		    }
		}
		int size = map.size();
		char[] result = new char[size];
		int[] isvisited = new int[26];
		for (char c: map.keySet()) {
			if (isvisited[c - 'a'] == 0 && !dfs(map, isvisited, c, result)) {
				return "";
			}
		}
		return String.valueOf(result);
    }
	
	public boolean dfs(HashMap<Character, Set<Character>> map, int[] isVisited, char c, 
			char[] result) {
		if (isVisited[c - 'a'] == 1) {
			return false;
		}
		isVisited[c - 'a'] = 1;
		for (char neighbor: map.get(c)) {
			if (isVisited[neighbor - 'a'] != 2 && !dfs(map, isVisited, neighbor, result)) {
				return false;
			}
		}
		isVisited[c - 'a'] = 2;
		result[result.length - order - 1] = c;
		order++;
		return true;
	}
	
	HashMap<Integer, Integer> inorderMap = new HashMap<>();
	public TreeNode buildTree(int[] preorder, int[] inorder) {
		if (preorder.length == 0) {
			return null;
		}
		for (int i = 0; i < inorder.length; ++i) {
			inorderMap.put(inorder[i], i);
		}
		return buildTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
	}
	
	public TreeNode buildTree(int[] preorder, int preStart, int preEnd,
			int[] inorder, int inStart, int inEnd) {
		if (preStart == preEnd) {
			return new TreeNode(preorder[preStart]); 
		}
		if (preStart > preEnd) {
			return null;
		}
		//left tree from inorderStart -> inorderMap.get(preorder[preStart]) - 1
		//preStart is the root, preStart + 1 ---> preStart + leftL - 1
		//pright: preEnd - rightL + 1 --> preEnd
		int rootPos = inorderMap.get(preorder[preStart]);
		int leftL = rootPos - inStart;
		int rightL = inEnd - rootPos;
		TreeNode root = new TreeNode(preorder[preStart]);
		TreeNode left = buildTree(preorder, preStart + 1, preStart + leftL, 
				inorder, inStart, rootPos - 1);
		TreeNode right = buildTree(preorder, preEnd- rightL + 1, preEnd, inorder,
				rootPos + 1, inEnd);
		root.left = left;
		root.right = right;
		return root;
	}
	
	
	public int candy(int[] ratings) {
		int n = ratings.length;
        int[] candies = new int[n];
        candies[0] = 1;
        for (int i = 1; i < n; i++) {
        	if (ratings[i] > ratings[i - 1]) {
        		candies[i] = candies[i - 1] + 1;
        	}
        	else if (ratings[i] == ratings[i - 1]) {
        		candies[i] = 1;
        	}
        	else if (ratings[i] < ratings[i - 1] && i!= n - 1
        			&& ratings[i] <= ratings[i + 1]){
        		candies[i] = 1;
        		int j = i - 1;
        		while (j >= 0 && ratings[j] > ratings[j + 1]) {
        			candies[j] = Math.max(candies[j], candies[j + 1] + 1);
        			j--;
        		}
        	}
        	else if (i == n - 1 && ratings[i] < ratings[i - 1]) {
        		candies[i] = 1;
        		int j = i - 1;
        		while (j >= 0 && ratings[j] > ratings[j + 1]) {
        			candies[j] = Math.max(candies[j], candies[j + 1] + 1);
        			j--;
        		}
        	}
        	
        }
        int sum = 0;
        for (int num: candies) {
        	sum += num;
        }
        System.out.println(Arrays.toString(candies));
        return sum;
    }
	
	public int[] singleNumber(int[] nums) {
		int is = 0;
        for (int num: nums) {
        	is ^= num;
        }
        int oneField = is &  (-is);
        int first = 0;
        int second = 0;
        for (int num: nums) {
        	if ((num & oneField) != 0) {
        		first ^= num;
        	}
        	else {
        		second ^= num;
        	}
        }
        return new int[] {first, second};
    }
	
	public void wallsAndGates(int[][] rooms) {
        int[] dx = new int[]{-1, 1, 0, 0};
        int[] dy = new int[]{0, 0, -1, 1};
        
        int n = rooms.length;
        int m = rooms[0].length;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (rooms[i][j] == 0) {
                    Queue<int[]> q = new LinkedList<>();
                    int step = 0;
                    q.offer(new int[]{i, j});
                    while (!q.isEmpty()) {
                    	int size = q.size();
                    	for (int count = 0; count < size; count++) {
                    		int[] point = q.poll();
                    		step++;
                    		for (int d = 0; d < 4; d++) {
                    			int nx = point[0] + dx[d];
                    			int ny = point[1] + dy[d];
                    			if (nx >=0 && nx < n && ny >= 0 && ny < m && 
                    					rooms[nx][ny] > 0) {
                    				rooms[nx][ny] = Math.min(step, rooms[nx][ny]);
                    				q.offer(new int[]{nx, ny});
                    			}
                    		}
                    	}
                    }
                }
            }
        }    
    }
	
	public int minTotalDistance(int[][] grid) {
		
        ArrayList<Integer> xs = new ArrayList<Integer> ();
        ArrayList<Integer> ys = new ArrayList<Integer> ();
        for (int i = 0; i < grid.length; ++i) {
        	for (int j = 0; j < grid[0].length; ++j) {
        		if (grid[i][j] == 1) {
        			xs.add(i);
        			ys.add(j);
        		}
        	}
        }
        Collections.sort(xs);
        Collections.sort(ys);
        if (xs.size() == 0 || grid.length == 0)
        	return 0;
        int dist = 0;
        int[] center = new int[] {xs.get(xs.size() / 2), ys.get(ys.size() / 2)};
        for (int i = 0; i < xs.size(); i++) {
        	dist += (Math.abs(xs.get(i) - center[0]) + Math.abs(ys.get(i) - center[1]));
        }
        return dist;
    }
	
	
	
	class PeekingIterator implements Iterator<Integer> {
		Iterator<Integer> iterator;
		Integer temp = null;

		public PeekingIterator(Iterator<Integer> iterator) {
			this.iterator = iterator;
			if (iterator.hasNext()) {
				temp = iterator.next();
			}
		}

	    // Returns the next element in the iteration without advancing the iterator.
		public Integer peek() {
	        return temp;
		}

		// hasNext() and next() should behave the same as in the Iterator interface.
		// Override them if needed.
		@Override
		public Integer next() {
		    Integer ret = temp;
		    if (iterator.hasNext()) {
		    	temp = iterator.next();
		    }
		    else {
		    	temp = null;
		    }
		    return ret;
		}

		@Override
		public boolean hasNext() {
		    return temp != null;
		}
	}
	
	public static class ValidWordAbbr {
	    HashSet<String> set = new HashSet<>();
	    
	    public ValidWordAbbr(String[] dictionary) {
	        for (String s: dictionary) {
	            String key = s.length() > 2? String.valueOf(s.charAt(0)) + (s.length() - 2) + s.charAt(s.length() - 1): s;
	            set.add(key);
	        }    
	    }

	    public boolean isUnique(String word) {
	        return set.contains(word);
	    }
	}
	
	public int countDigitOne(int n) {
        if (n <= 0) {
            return 0;
        }
        int countD = 1;
        while (n / countD >= 10) {
            countD *= 10;
        }
        int ones = 0;
        while (countD > 0) {
            int d = n / countD;
            if (d % 10 != 1) {
                ones += (((d - 1) / 10) + 1) * countD;
            }
            else {
                ones += (d / 10) * countD + n % countD + 1;
            }
            countD /= 10;
        }
        return ones;
        
    }
	
	
	public List<Integer> closestKValues(TreeNode root, double target, int k) {
        List<Integer> rv = new ArrayList<>();
        Stack<TreeNode> s1 = new Stack<>();
        Stack<TreeNode> s2 = new Stack<>();
        List<Integer> successors = new ArrayList<>();
        List<Integer> predessors = new ArrayList<>();
        //find the successor
        TreeNode successor = root;
        while (successors.size() < k && (successor != null || !s1.isEmpty())) {
            if (successor != null){
                if (successor.val < target) {
                    successor = successor.right;
                }
                else {
                    s1.push(successor);
                    successor = successor.left;
                }
            }
            else {
                successor = s1.pop();
                successors.add(successor.val);
                successor = successor.right;
            }
        }
        TreeNode predessor = root;
        while (predessors.size() < k && (predessor != null || !s2.isEmpty())) {
            if (predessor != null) {
                if (predessor.val > target) {
                    predessor = predessor.left;
                }
                else {
                    s2.push(predessor);
                    predessor = predessor.right;
                }
            }
            else {
                predessor = s2.pop();
                predessors.add(predessor.val);
                predessor = predessor.left;
            }
        }
        int i = 0;
        int j = 0;
        while (i < predessors.size() && j < successors.size()) {
        	double diff1 = Math.abs(predessors.get(i) - target);
        	double diff2 = Math.abs(successors.get(j) - target);
        	if (diff1 < diff2) {
        		rv.add(predessors.get(i++));
        	}
        	else {
        		rv.add(successors.get(j++));
        	}
        }
        while (i < predessors.size()) {
        	rv.add(predessors.get(i++));
        }
        while (i < successors.size()) {
        	rv.add(successors.get(j++));
        }
        return rv;
    }
	
//	public static class Codec {
//
//	    // Encodes a list of strings to a single string.
//	    public String encode(List<String> strs) {
//	        StringBuilder header = new StringBuilder();
//	        StringBuilder whole = new StringBuilder();
//	        for (String str: strs) {
//	            whole.append(str);
//	            header.append(String.valueOf(str.length()) + " ");
//	        }
//	        if (header.length() > 0) {
//	            header.deleteCharAt(header.length() - 1);
//	        }
//	        return header.toString() + "#" + whole.toString();
//	    }
//
//	    // Decodes a single string to a list of strings.
//	    public List<String> decode(String s) {
//	        int i = 0;
//	        List<String> list = new ArrayList<>();
//	        while (i < s.length() && s.charAt(i) != '#'){
//	            i++;
//	        }
//	        String[] header=  s.substring(0, i).split(" ");
//	        String self = s.substring(i + 1);
//	        if (self.length() == 0) {
//	        	return list;
//	        }
//	        int count = 0;
//	        for (i = 0; i < header.length; ++i) {
//	            int length = Integer.parseInt(header[i]);
//	            list.add(self.substring(count, count + length));
//	            count += length;
//	        }
//	        return list;
//	    }
//	}
	
	public static class MedianFinder {
	    PriorityQueue<Integer> lowerBracket = new PriorityQueue<>(Collections.reverseOrder());
	    PriorityQueue<Integer> upperBracket = new PriorityQueue<>();
	    double medium;
	    // Adds a number into the data structure.
	    public void addNum(int num) {
	        if (lowerBracket.size() == 0 && upperBracket.size() == 0){
	            lowerBracket.offer(num);
	            medium = num;
	            return;
	        }
	        if (num <= medium) {
	            lowerBracket.offer(num);
	            if (lowerBracket.size() - upperBracket.size() > 1) {
		            upperBracket.offer(lowerBracket.poll());
		        }
	        }
	        else {
	            upperBracket.offer(num);
	            if (upperBracket.size() > lowerBracket.size()) {
	            	lowerBracket.offer(upperBracket.poll());
	            }
	        }
	        if (lowerBracket.size() - upperBracket.size() == 1){
	            medium = lowerBracket.peek();
	        }
	        else {
	            medium = lowerBracket.peek() + 0.5 * (upperBracket.peek() - lowerBracket.peek());
	        }
	        
	    }

	    // Returns the median of current data stream
	    public double findMedian() {
	        return medium;
	    }
	};
	
	public List<String> addOperators(String num, int target) {
        ArrayList<String> rv = new ArrayList<>();
        for (int i = 1; i <= num.length(); i++) {
        	long sub = Long.parseLong(num.substring(0, i));
        	if (num.charAt(0) == '0' && i > 1) {
        		break;
        	}
        	addDFS(num, target, i, rv, num.substring(0, i), sub, sub);
        }
        return rv;
    }
    
    public void addDFS(String num, int target, int pos, List<String> list, String prefix, long current, long lastNumber) {
        //System.out.println(prefix + " value = " + current);
    	if (pos == num.length()) {
            if (current == target)
                list.add(prefix);
            return;
        } 
        for (int i = pos + 1; i <= num.length(); i++) {
            //put add
        	if (num.charAt(pos) == '0' && i == pos + 2) {
        		return;
        	}
            long sub = Long.parseLong(num.substring(pos, i));
            addDFS(num, target, i, list, prefix + "+" + sub, current + sub, sub);
            addDFS(num, target, i, list, prefix + "-" + sub, current - sub, -sub);
            if (pos != 0)
                addDFS(num, target, i, list, prefix + "*" + sub, current - lastNumber + lastNumber * sub, lastNumber * sub);
        }
    }
	
	public int minMeetingRooms(Interval[] intervals) {
        Arrays.sort(intervals, new Comparator<Interval> (){
           public int compare(Interval o1, Interval o2) {
               if (o1.start != o2.start)
                    return o1.start - o2.start;
                return o1.end - o2.end;
           } 
        });
        PriorityQueue<Interval> meeting = new PriorityQueue<>(new Comparator<Interval>(){
            public int compare(Interval o1, Interval o2) {
                if (o1.end != o2.end)
                    return o1.end - o2.end;
                return o1.start - o2.start;
            }
        });
        
        int i = 0;
        int max = 0;
        while (i < intervals.length) {
            if (meeting.isEmpty() || meeting.peek().end > intervals[i].start) {
                 meeting.offer(intervals[i++]);
                 max = Math.max(max, meeting.size());
            }
            else {
                meeting.poll();
            }
        }
        return max;
    }
	
	
	public String minWindow(String s, String t) {
        int[] count_t = new int[128];
        for (int j = 0; j < t.length(); ++j) {
            count_t[t.charAt(j)]++;
        }
        int count = t.length();
        String min = "";
        int i = 0, j = 0;
        while (i < s.length()) {
            if (count_t[s.charAt(i++)]-- > 0) count--;
            while (count == 0) {
                min = (min.length() == 0 || min.length() > (i - j))? s.substring(j, i): min;
                if (count_t[s.charAt(j++)]++ >= 0) count++;
            }
        }
        return min;
    }
	
	public List<List<Integer>> getFactors(int n) {
		List<List<Integer>> rv = new ArrayList<List<Integer>>();
		ArrayList<Integer> list = new ArrayList<Integer> ();
		dfs(2, n, rv, list);
		return rv;
    }
	
	public void dfs(int base, int n, List<List<Integer>> rv, ArrayList<Integer> current) {
		for (int i = base; i * i <= n; i++) {
			if (n % i == 0) {
				current.add(i);
				dfs(i, n / i, rv, current);
				ArrayList<Integer> copy = new ArrayList<>();
				copy.addAll(current);
				copy.add(n / i);
				rv.add(copy);
				current.remove(current.size() - 1);
			}
		}
	}
	
	public static class Codec {

	    // Encodes a tree to a single string.
	    public String serialize(TreeNode root) {
	        StringBuilder sb = new StringBuilder();
	        serialize(root, sb);
	        return sb.toString();
	    }
	    
	    public void serialize(TreeNode root, StringBuilder sb) {
	        if (root == null) {
	        	//System.out.println("#");
	            sb.append("# ");
	            return;
	        }
	        else {
	            sb.append(String.valueOf(root.val) + " ");
	        }
	        //System.out.println(root.val + " verify");
	        serialize(root.left, sb);
	        serialize(root.right, sb);
	    }

	    // Decodes your encoded data to tree.
	    public TreeNode deserialize(String data) {
	        String[] nodes = data.split(" ");
	        if (nodes[0].equals("#"))   return null;
	        TreeNode rt = new TreeNode(0);
	        parse(nodes, 0, rt);
	        return rt;
	    }
	    
	    public int parse(String[] nodes, int position, TreeNode root) {
	        root.val = Integer.parseInt(nodes[position]);
	        int offset = 1;
	        if (!nodes[position + offset].equals("#")) {
	            TreeNode left = new TreeNode(0);
	            root.left = left;
	            offset += parse(nodes, position + 1, left);
	        }
	        else {
	            offset = 2;
	        }
	        if (!nodes[position + offset].equals("#")) {
	            TreeNode right = new TreeNode(0);
	            root.right = right;
	            offset += parse(nodes, position + offset, right);
	        }
	        else {
	            offset += 1;
	        }
	        return offset;
	    }
	}
	
	public static void main(String[] args) {
		Solution_251_to_300 s = new Solution_251_to_300();
		//Codec c = new Codec();
		MedianFinder mf = new MedianFinder();
		mf.addNum(1);
		System.out.println(mf.medium);
		mf.addNum(2);
		System.out.println(mf.medium);
		mf.addNum(4);
		System.out.println(mf.medium);
		mf.addNum(-3);
		System.out.println(mf.medium);
		mf.addNum(0);
		System.out.println(mf.medium);
		mf.addNum(9);
		System.out.println(mf.medium);
		
		List<String> list = new ArrayList<> ();
//		list.add(" #apple");
//		list.add("orange");
		String[] test = new String[] {"wrt"};
		TreeNode node = new TreeNode(1);
		TreeNode node2 = new TreeNode(2);
		TreeNode node3 = new TreeNode(3);
		TreeNode node4 = new TreeNode(4);
		TreeNode node5 = new TreeNode(5);
		node.left = node2;
		node.right = node3;
		node3.left = node4;
		node4.right = node5;
		Codec c = new Codec();
		System.out.println(c.deserialize(c.serialize(node)).right.val);

	}

}
