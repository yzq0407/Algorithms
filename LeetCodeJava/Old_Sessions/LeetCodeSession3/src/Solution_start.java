import java.util.*;

public class Solution_start {
	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len = nums1.length + nums2.length;
        if (len % 2 == 0)
            return ((double) findKthSortedArrays(nums1, nums2, 0, nums1.length - 1, 0, nums2.length - 1, len / 2 - 1)
                + (double) findKthSortedArrays(nums1, nums2, 0, nums1.length - 1, 0, nums2.length - 1, len / 2)) / 2;
        else
            return (double) findKthSortedArrays(nums1, nums2, 0, nums1.length - 1, 0, nums2.length - 1, len / 2);
    }
    
    public int findKthSortedArrays(int[] nums1, int[] nums2, int s1, int e1, int s2, int e2, int k) {
        if (s1 > e1)    return nums2[s2 + k];
        if (s2 > e2)    return nums1[s1 + k];
        if (k == 0)     return Math.min(nums1[s1], nums2[s2]);
        int mid1 = s1 + (e1 - s1) / 2;
        int mid2 = s2 + (e2 - s2) / 2;
        if (nums1[mid1] < nums2[mid2]) {
            //smaller thean nums2[mid2] is mid1 - s1 + 1 + mid2 - s2
            int smaller = mid1 - s1 + 1 + mid2 - s2;
            if (smaller <= k) {
                //discard all this
                return findKthSortedArrays(nums1, nums2, mid1 + 1, e1, s2, e2, k - (mid1 - s1 + 1));
            }
            else {
                return findKthSortedArrays(nums1, nums2, s1, e1, s2, mid2 - 1, k);
            }
        }
        else {
            int smaller = mid2 - s2 + 1 + mid1 - s1;
            if (smaller <= k) {
                return findKthSortedArrays(nums1, nums2, s1, e1, mid2 + 1, e2, k - (mid2 - s2 + 1));
            }
            else {
                return findKthSortedArrays(nums1, nums2, s1, mid1 - 1, s2, e2, k);
            }
        }
    }
    
    public boolean isMatch(String s, String p) {
        int n = s.length();
        int m = p.length();
        boolean[][] dp = new boolean[n + 1][m + 1];
        dp[0][0] = true;
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                //trying to match s[0: i] to p[0: j]
                //if (s[i - 1] == p[j - 1] || p[j - 1] == '.') match s[0: i -1] and p[0: j - 1]
                //if (p[j - 1] == '*') match s[0: i - 1] and p[0: j] or s[0: i - 1] and p [0 : j - 1] or s[0: i] and p[0: j-1]
                if (i == 0 && j == 0)   continue;
                if (j == 0)   continue;
                if (i == 0) {
                    dp[i][j] = p.charAt(j - 1) == '*' && dp[i][j - 2];
                    continue;
                }
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else if (p.charAt(j - 1) == '*') {
                	if (s.charAt(i - 1) == p.charAt(j - 2)){
                		dp[i][j] = dp[i - 1][j - 2] || dp[i][j - 2] || dp[i - 1][j];
                	}
                	else {
                		dp[i][j] = dp[i][j - 2];
                	}
                }
            }
            //System.out.println(Arrays.toString(dp[i]));
        }
        return dp[n][m];
    }
    
    public ArrayList<ArrayList<Integer>> buildingOutline(int[][] buildings) {
    	ArrayList<ArrayList<Integer>> res = new ArrayList<>();
    	if (buildings.length == 0)	return res;
    	ArrayList<int[]> cps = new ArrayList<>();
    	for (int i = 0; i < buildings.length; ++i) {
    		cps.add(new int[]{i, 0});
    		cps.add(new int[]{i, 1});
    	}
    	Collections.sort(cps, new Comparator<int[]>() {
			@Override
			public int compare(int[] a, int[] b) {
				// TODO Auto-generated method stub
				if(buildings[a[0]][a[1]] != buildings[b[0]][b[1]])	
	    			return buildings[a[0]][a[1]] - buildings[b[0]][b[1]];
	    		if (a[1] != b[1])	return a[1] - b[1];
	    		if (a[1] == 0)	return buildings[b[0]][2] - buildings[a[0]][2];
	    		return buildings[a[0]][2] - buildings[b[0]][2];
			}		
    	});
    	
    	int left = 0;
    	TreeSet<Integer> set = new TreeSet<Integer>(new Comparator<Integer>(){

			@Override
			public int compare(Integer a, Integer b) {
				// TODO Auto-generated method stub
				if (buildings[a][2] != buildings[b][2]) return buildings[a][2] - buildings[b][2];
				return a - b;
			}
    	});
    	
    	for (int[] cp: cps) {
    		int idx = cp[0], height = buildings[idx][2];
    		//System.out.println("critical point " + buildings[idx][cp[1]] + (cp[1] == 0? " enter ": "exit") + " with height: " + height);
    		if (cp[1] == 0) {
    			if (set.size() == 0)	left = buildings[idx][0];
    			else if (buildings[set.last()][2] < height) {
    				ArrayList<Integer> building = new ArrayList<>();
    				building.add(left);
    				building.add(buildings[idx][0]);
    				building.add(buildings[set.last()][2]);
    				res.add(building);
    				left = buildings[idx][0];
    			}
    			set.add(idx);
    		}
    		else {
    			if (!set.remove(idx))	System.out.println("didn't find it !!");
    			int currentHeight = set.size() == 0? 0: buildings[set.last()][2];
    			
    			if (height > currentHeight) {
    				//System.out.println(" max building height: " + buildings[set.last()][2]);
    				ArrayList<Integer> building = new ArrayList<>();
    				building.add(left);
    				building.add(buildings[idx][1]);
    				building.add(buildings[idx][2]);
    				res.add(building);
    				left = buildings[idx][1];
    			}
    		}
    	}
    	return res;
    }


    public String shortestPalindrome(String s) {
        //the longest equal prefix s[0:next[i]] == s[i - next[i]: i]
        if (isPalindrome(s))    return s;
        String str = s + "#" + new StringBuilder(s).reverse().toString();
        int[] next = new int[str.length()];
        next[0] = -1;
        for (int i = 1; i < str.length(); i++) {
            int j = i - 1;
            while (j != -1 && str.charAt(i) != str.charAt(next[j] + 1)) {
                j = next[j];
            }
            if (j == -1)    next[i] = -1;
            else next[i] = next[j] + 1;
        }
        int palinLen = next[str.length() - 1] + 1;
        return new StringBuilder(s.substring(palinLen)).reverse().toString() + s;
    }

    public boolean isPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j--)) return false;
        }
        return true;
    }

    public int strStr(String haystack, String needle) {
        int[] next = getNext(needle);
        int i = 0;
        int j = 0;
        while (i < haystack.length() && j < needle.length()){
            if (j == -1 || haystack.charAt(i) == needle.charAt(j)) {
                i++;
                j++;
            }
            else {
                j = next[j];
            }
        }
        if (j == needle.length())   return i - needle.length();
        return -1;
    }

    public int[] getNext(String needle) {
        int[] next =new int[needle.length()];
        next[0] = -1;
        for (int i = 1; i < needle.length(); i++) {
            //we know that i - 1
            int j = i - 1;
            while (next[j] != -1 && needle.charAt(next[j]) != needle.charAt(i - 1)) {
                j = next[j];
            }
            next[i] = next[j] + 1;
        }
        return next;
    }

    public List<int[]> getSkyline(int[][] buildings) {
        int len = buildings.length;
        int[][] cps = new int[len * 2][3];
        //each cp contains cp[0] == x axis, cp[1] in(0) or out(1)?  cp[3] height
        for (int i = 0; i < len; i++) {
            int[] building = buildings[i];
            //enter point
            cps[i * 2][0] = building[0];
            cps[i * 2][1] = 0;
            cps[i * 2][2] = building[2];
            cps[i * 2 + 1][0] = building[1];
            cps[i * 2 + 1][1] = 1;
            cps[i * 2 + 1][2] = building[2];
        }
        Arrays.sort(cps, (a, b) -> {
            if (a[0] != b[0])   return a[0] - b[0];
            if (a[1] != b[1])   return a[1] - b[1];
            if (a[1] == 0)  return b[2] - a[2];
            else    return a[2] - b[2];
        });
        List<int[]> rv = new ArrayList<>();
        TreeMap<Integer, Integer> map = new TreeMap<>();
        //height -> number of heights
        for (int[] cp: cps) {
            if (cp[1] == 0) {
                if (map.isEmpty() || map.lastKey() < cp[2]) {
                    rv.add(new int[]{cp[0], cp[2]});
                }
                if (map.containsKey(cp[2])) map.put(cp[2], map.get(cp[2]) + 1);
                else map.put(cp[2], 1);
            }
            //this is an exit
            else {
                if (map.get(cp[2]) == 1)    map.remove(cp[2]);
                else    map.put(cp[2], map.get(cp[2]) - 1);
                if (map.isEmpty() || map.lastKey() < cp[2]) {
                    rv.add(new int[]{cp[0], map.isEmpty()? 0: map.lastKey()});
                }
            }
        }
        return rv;
    }

    public int countDigitOne(int n) {
        int pos = 1;
        int sum = 0;
        while (n / pos != 0) {
            int head = n / pos;
            if (head % 10 == 0) {
                sum += head / 10 * pos;
            }
            else if (head % 10 == 1) {
                sum += head / 10 * pos;
                sum += (n % pos + 1);
            }
            else {
                sum += (head / 10 + 1) * pos;
            }
            pos *= 10;
        }
        return sum;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> dq = new LinkedList<Integer> ();
        //maitain a decreasing sequence of numbers with length less than or equal to k
        int[] rv = new int[nums.length - k + 1];
        for (int i = 0; i < nums.length; i++) {
            //add this number into the deque
            while (!dq.isEmpty() && nums[dq.peekLast()] < nums[i]) dq.pollLast();
            dq.offerLast(i);
            if (i - dq.peekFirst() == k - 1)    dq.pollFirst();
            if (i >= k) rv[i - k + 1] = nums[dq.peekFirst()];
        }
        return rv;
    }

    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> rv = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        dfs(n, 2, rv, list);
        return rv;
    }

    public void dfs(int n, int lastFactor, List<List<Integer>> rv, List<Integer> list) {
        for (int i = lastFactor; n / i >= i; i++) {
            if (n % i == 0) {
                list.add(i);
                ArrayList<Integer> copy = new ArrayList<>();
                copy.addAll(list);
                copy.add(n / i);
                rv.add(copy);
                dfs(n / i, i, rv, list);
                list.remove(list.size() - 1);
            }
        }
    }

    public int nthUglyNumber(int n) {
        int[] dp = new int[n];
        dp[0] = 1;
        int p2 = 0;
        int p3 = 0;
        int p5 = 0;
        for (int i = 1; i < n; i++) {
            dp[i] = Math.min(dp[p2] * 2, Math.min(dp[p3] * 3, dp[p5] * 5));
            if (dp[i] == dp[p2] * 2)    p2++;
            if (dp[i] == dp[p3] * 3)    p3++;
            if (dp[i] == dp[p5] * 5)    p5++;

        }

        return dp[n - 1];
    }

    public String alienOrder(String[] words) {
        if (words.length == 0)    return "";
        Map<Character, Set<Character>> map = new HashMap<>();
        for (int i = 0; i < words.length - 1; i++) {
            String current = words[i];
            String next = words[i + 1];
            int j = 0;
            for (j = 0; j < Math.min(current.length(), next.length()); j++) {
                char c = current.charAt(j);
                char n = next.charAt(j);
                map.computeIfAbsent(c, k -> new HashSet<Character> ());
                map.computeIfAbsent(n, k -> new HashSet<Character> ());
                if (c != n) {
                    map.get(c).add(n);
                    break;
                }
            }
            while (j < current.length())    map.computeIfAbsent(current.charAt(j++), k -> new HashSet<Character>());
        }
        for (int j = 0; j < words[words.length - 1].length(); j++)   map.computeIfAbsent(words[words.length - 1].charAt(j), k -> new HashSet<Character>());
        // System.out.println(map);
        int[] visited = new int[26];
        StringBuilder sb = new StringBuilder();
        for (Character current: map.keySet()) {
            if (visited[current - 'a'] == 0 && !tpSort(map, current, visited, sb)) return "";
        }
        return sb.toString();

    }

    public boolean tpSort(Map<Character, Set<Character>> map, char current, int[] visited, StringBuilder sb) {
        visited[current - 'a'] = -1;
        for (Character neighbor: map.get(current)) {
            if (visited[neighbor - 'a'] == -1 || (visited[neighbor - 'a'] == 0 && !tpSort(map, neighbor, visited, sb)))  return false;
        }
        visited[current - 'a'] = 1;
        sb.insert(0, current);
        return true;
    }

    public List<String> findItinerary(String[][] tickets) {
        Map<String, PriorityQueue<String>> map = new HashMap<> ();
        for (String[] ticket: tickets) {
            map.computeIfAbsent(ticket[0], k -> new PriorityQueue<>()).offer(ticket[1]);
            map.computeIfAbsent(ticket[1], k -> new PriorityQueue<>());
        }
        List<String> list = new ArrayList<> ();
        dfs(list, map, "JFK");
        return list;
    }

    public void dfs(List<String> list, Map<String, PriorityQueue<String>> map, String current) {
        while (!map.get(current).isEmpty())
            dfs(list, map, map.get(current).poll());
        list.add(0, current);
    }
    
  
    
    public int hIndex(int[] citations) {
        if (citations.length == 0)   return 0;
        //hIndex citations[hIndex] == len - hIndex + 1
        int low = 0;
        int high = citations.length - 1;
        while (low <= high) {
            //citations.length - high <= citations[high]
            int mid = low + (high - low) / 2;
            int greater = citations.length - mid;
            if (greater <= citations[mid])  high = mid - 1;
            else    low = mid + 1;
        }
        return citations.length - (high + 1);
    }

    public List<String> addOperators(String num, int target) {
        //00, 05,
        //when we start with 0, any number except 0 itself is not allowed
        //keep a current result, when we next number, choose to + or -
        //but if we choose to *, have to subtract the last number and then do result - last number + last number * new number
        //then update last number into last number * new number
        List<String> rv = new ArrayList<>();
        dfs(rv, 0, 0, num, 0, "", target);
        return rv;
    }

    public void dfs(List<String> rv, int current, int last, String num, int pos, String clause, int target) {
        if (pos == num.length() && current == target) {
            rv.add(clause.substring(1));
            return;
        }
        for (int i = pos + 1; i <= num.length(); i++) {
            if (num.charAt(pos) == '0' && i - pos > 1)  return;
            String sub = num.substring(pos, i);
            if (Long.parseLong(sub) > Integer.MAX_VALUE)    return;
            int n = Integer.parseInt(sub);
            //three scenarios
            // add
            //sb.append("+");
            if (current + n - n == current)
                dfs(rv, current + n, n, num, i, clause + "+" + sub, target);
            if (pos == 0)   continue;
            if (current - n + n == current)
                dfs(rv, current - n, -n, num, i, clause + "-" + sub, target);
            if (current - last + last * n - last * n == current - last)
                dfs(rv, current - last + last * n, last * n, num, i, clause + "*" + sub, target);
        }
    }

    public void wallsAndGates(int[][] rooms) {
        if (rooms.length == 0)  return;
        int n = rooms.length;
        int m = rooms[0].length;
        int[] dx = new int[]{-1, 1, 0, 0};
        int[] dy = new int[]{0, 0, -1, 1};
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (rooms[i][j] == 0)   queue.offer(new int[]{i, j});
            }
        }
        int step = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] point = queue.poll();
                int x = point[0], y = point[1];
                //rooms[x][y] = step;
                for (int d = 0; d < 4; ++d) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    if (nx < n && nx >= 0 && ny < m && ny >= 0 && rooms[nx][ny] == Integer.MAX_VALUE) {
                        rooms[nx][ny] = step;
                        queue.offer(new int[]{nx, ny});
                    }
                }
            }
            step++;
        }
    }

    public List<String> removeInvalidParentheses(String s) {
        List<String> rv =new ArrayList<>();
        dfs(s, 0, new char[]{'(', ')'}, rv);
        return rv;
    }

    public void dfs(String s, int last, char[] tokens, List<String> rv) {
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == tokens[0])   count++;
            else if (s.charAt(i) == tokens[1])   count--;
            if (count < 0) {
                for (int j = last; j <= i; j++) {
                    if (s.charAt(j) == tokens[1] && (j == last || s.charAt(j - 1) != tokens[1]))
                        dfs(s.substring(0, j) + s.substring(j + 1), j, tokens, rv);
                }
                return;
            }
        }
        String reverse = new StringBuilder(s).reverse().toString();
        if (tokens[0] == ')')   rv.add(reverse);
        else {
            dfs(reverse, 0, new char[]{tokens[1], tokens[0]}, rv);
        }
    }

    public class NumMatrix {
        int[][] matrix;
        int[][] rangeSum;
        int n, m;

        public NumMatrix(int[][] val) {
            n = val.length;
            m = val[0].length;
            matrix = new int[n][m];
            rangeSum = new int[n + 1][m + 1];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    update(i, j, val[i][j]);
                }
            }
        }

        public void update(int row, int col, int val) {
            int diff = val - matrix[row][col];
            matrix[row][col] += diff;
            for (int i = row + 1; i <= n; i += (i & -i)) {
                for (int j = col + 1; j <= m; j += (j & -j)) {
                    rangeSum[i][j] += diff;
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            return sumPoint(row2, col2) + sumPoint(row1 - 1, col1 - 1) 
                - sumPoint(row1 - 1, col2) - sumPoint(row2, col1 - 1); 
        }

        private int sumPoint(int row, int col) {
            int sum = 0;
            for (int i = row + 1; i > 0; i -= (i & -i)) {
                for (int j = col + 1; j > 0; j -= (j & -j)) {
                    sum += rangeSum[i][j];
                }
            }
            return sum;
        }
    }
    
    public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        int max = 0;
        for (int i = 1; i < s.length(); ++i) {
            if (s.charAt(i) == '(') continue;
            if (s.charAt(i - 1) == '(') dp[i] = 2 + (i >= 2? dp[i - 2]: 0);
            else if (i - 1 - dp[i - 1] >= 0 && s.charAt(i - 1 - dp[i - 1]) == '('){
                dp[i] = dp[i - 1] + 2;
                if (i - 1 - dp[i - 1] > 0)  dp[i] += dp[i - 2 - dp[i - 1]];
            }
            max = Math.max(dp[i], max);
        }
        return max;
    }
    
    public int maximumGap(int[] nums) {
        if (nums.length < 2)    return 0;
        long max = Integer.MIN_VALUE, min = Integer.MAX_VALUE;
        for (int num: nums) {
            max = Math.max(max, num);
            min = Math.min(min, num);
        }
        if (max == min) return 0;
        //
        long gap = (max - min) % (nums.length - 1) == 0? (max - min) / (nums.length - 1): (max - min) / (nums.length - 1) + 1;
        int[][] buckets = new int[2][(int)(((long)max - (long)min) / gap) + 1];
        Arrays.fill(buckets[0], Integer.MIN_VALUE);
        Arrays.fill(buckets[1], Integer.MAX_VALUE);
        for (int num: nums) {
            int bucket = (int)((((long) num)  - min)/ gap);
            buckets[0][bucket] = Math.max(num, buckets[0][bucket]);
            buckets[1][bucket] = Math.min(num, buckets[1][bucket]);
        }
        int maxGap = 0, prevMax = Integer.MIN_VALUE;
        for (int i = 0; i < buckets[0].length; i++) {
            if (buckets[0][i] == Integer.MIN_VALUE)  continue;
            if (prevMax != Integer.MIN_VALUE) {
                maxGap = Math.max(buckets[1][i] - prevMax, maxGap);
            }
            prevMax = buckets[0][i];
        }
        return maxGap;
    }
    
    
    public int candy(int[] ratings) {
        int[] dp = new int[ratings.length];
        dp[0] = 1;
        for (int i = 1; i < ratings.length; ++i) {
            if (ratings[i] > ratings[i - 1])    {
                fix(dp, i - 1, ratings);
                dp[i] = dp[i - 1] + 1;
            }
            else if (ratings[i] == ratings[i - 1]) {
                fix(dp, i - 1, ratings);
                dp[i] = 1;
            }
            else dp[i] = Math.min(dp[i - 1] - 1, 1);
        }
        fix(dp, ratings.length - 1, ratings);
        //System.out.println(Arrays.toString(dp));
        int sum = 0;
        for (int i = 0; i < ratings.length; ++i){
        	System.out.println("ratings: "+ ratings[i] + " candies: "+ dp[i]);
            sum += dp[i];
        }
        return sum;
    }
    
    public void fix(int[] dp, int idx, int[] ratings) {
        if (dp[idx] > 0) return;
        dp[idx--] = 1;
        while (idx >= 0 && ratings[idx] > ratings[idx + 1]) {
            dp[idx] = Math.max(dp[idx + 1] + 1, dp[idx]);
            idx--;
        }
    }

    public static void main(String[] args){
    	Solution_start s = new Solution_start();
        int[][] buildings = new int[][]{{63,96,72},{25,32,201},{47,70,247},{19,39,25},{61,77,181},{36,72,264},{35,42,211},{15,48,118},{29,77,144},{67,100,245},{75,86,91},{37,90,180},{57,72,38},{56,100,14},{24,50,119},{48,94,112},{5,87,229},{38,52,42},{58,84,173},{6,66,119},{27,77,218},{53,86,200},{45,65,201},{32,92,121},{45,97,263},{39,64,50},{58,72,204},{61,62,158},{11,64,151},{22,96,241},{17,64,2},{49,100,204},{5,33,99},{29,91,227},{43,82,8},{13,90,93},{47,51,39},{59,89,104},{14,38,197},{8,72,266},{37,81,106},{5,40,205},{59,73,253},{12,59,217},{38,67,70},{53,76,30},{24,71,215},{2,39,245},{4,10,209},{51,86,211},{14,55,242},{20,51,107},{6,36,169},{42,92,58},{24,70,7},{1,100,81},{8,15,14},{3,33,46},{6,77,34},{37,50,228},{62,68,150},{46,56,127},{31,66,85},{34,39,8},{77,81,6},{22,63,21},{8,83,221},{8,66,166},{37,49,215},{30,56,95},{50,87,130},{56,57,13},{92,97,86},{13,32,266},{75,83,82},{55,93,38},{57,64,135},{49,82,162},{7,16,179},{5,77,136},{8,93,118},{2,98,8},{7,40,207},{62,78,200},{25,38,91},{25,76,243},{54,77,110},{20,34,265},{51,61,40},{7,63,218}};
        // for (int[] building: s.getSkyline(buildings)) {
        //     System.out.println(Arrays.toString(building));
        // }
        String[] dict = new String[] {"wrt", "wrf", "er", "ett", "rftt"};
        System.out.println(s.candy(new int[]{58,21,72,77,48,9,38,71,68,77,82,47,25,94,89,54,26,54,54,99,64,71,76,63,81,82,60,64,29,51,87,87,72,12,16,20,21,54,43,41,83,77,41,61,72,82,15,50,36,69,49,53,92,77,16,73,12,28,37,41,79,25,80,3,37,48,23,10,55,19,51,38,96,92,99,68,75,14,18,63,35,19,68,28,49,36,53,61,64,91,2,43,68,34,46,57,82,22,67,89}));
        //System.out.println(s.getSkyline(buildings));
    }
}
