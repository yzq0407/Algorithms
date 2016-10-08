import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;



public class Solutions {
	
	public List<List<Integer>> permuteUnique(int[] nums) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
    	LinkedList<Integer> list = new LinkedList<Integer> ();
    	HashMap<Integer, Integer> isVisited = new HashMap<Integer, Integer>();
    	for (int i:nums){
    		if (isVisited.containsKey(i)){
    			isVisited.put(i, isVisited.get(i)+1);
    		}
    		else
    			isVisited.put(i, 1);
    	}
    	permuteUnique(nums, isVisited, result, list);
    	return result;
        
    }
	class Pair {
		int x;
		int y;
		Pair (int x, int y) {
			this.x = x;
			this.y = y;
		}
	}
	
	public List<String> restoreIpAddresses(String s) {
        HashMap<String, List<String>> memo = new HashMap<>();
        return restoreIPAddresses(s, 4, memo);
    }
	
	public List<String> restoreIPAddresses(String sb, int seg,
			HashMap<String, List<String>> memo) {
		String tag = sb + " " + seg;
		if (memo.containsKey(tag)) 
			return memo.get(tag);
		List<String> ret = new ArrayList<>();
		if (sb==null || sb.length()>seg*3 || sb.length()<seg) {
			memo.put(tag, ret);
			return ret;
		}
		if (seg==1 && Integer.parseInt(sb)<256 && !(sb.length()>1&& sb.charAt(0)=='0')) {
			ret.add(sb);
			memo.put(tag, ret);
			return ret;
		}
		for (int i = 1; i<=Math.min(3, sb.length()); i++) {
			String segment = sb.substring(0, i);
			if (Integer.valueOf(segment)<256 && !(i>1&& segment.charAt(0)=='0'))
			{
				List<String> children = restoreIPAddresses(sb.substring(i), seg-1, memo);
				for (String child: children) {
					ret.add(segment+ "." + child);
				}
			}
		}
		memo.put(tag, ret);
		return ret;
	}
	
	public void solveSudoku(char[][] board) {
        //preprocess
		solve (board, 0, 0);
    }
	
	int maxCount = 0;
    public int numberOfPatterns(int m, int n) {
        int[] board = new int[9];
        for (int i = 0; i < 9; i++){
        	dfs(board, m, n, i, 0);
        }
        return maxCount;
    }
    
    class Coordinate {
    	int x;
    	int y;
    	Coordinate(int x, int y) {
    		this.x = x;
    		this.y = y;
    	}
    }
    
    public int shortestDistance(int[][] grid) {
        int[][] distance = new int[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
        	for (int j = 0; j < grid[0].length; j++) {
        		if (grid[i][j] == 1)
        			bfs(grid, distance, i, j);
        	}
        }
        int s = Integer.MAX_VALUE;
        for (int i = 0; i < distance.length; i++) {
        	for (int j = 0; j < grid[0].length; j++) {
        	    if (grid[i][j] == 0 && distance[i][j]!= 0)
        		    s = Math.min(s, distance[i][j]);
        	}
        }
        return s == Integer.MAX_VALUE? -1 : s;
    }
    
    public void bfs(int[][] grid, int[][] distance, int i, int j){
    	boolean[][] isVisited = new boolean[distance.length][distance[0].length];
    	isVisited[i][j] = true;
    	int[] dx = new int[] {-1, 1, 0, 0};
    	int[] dy = new int[] {0, 0, -1, 1};
    	Queue<Coordinate> q = new LinkedList<>();
    	q.offer(new Coordinate(i, j));
    	int dist = 0;
    	while (!q.isEmpty()) {
    		int size = q.size();
    		for (int count = 0; count < size; count++) {
    			Coordinate c = q.poll();
    			distance[c.x][c.y] += dist;
    			for (int direc = 0; direc < 4; direc++) {
    				int nx = c.x + dx[direc];
    				int ny = c.y + dy[direc];
    				if (nx >= 0&& nx < distance.length && ny >= 0 && ny < distance[0].length
    						&& !isVisited[nx][ny] && grid[nx][ny] == 0) {
    					q.offer(new Coordinate(nx, ny));
    					isVisited[nx][ny] = true;
    				}
    			}
    		}
    		dist++;
    	}
    	for (int x = 0; x < distance.length; x++) {
    		for (int y = 0; y < distance[0].length; y++) {
    			if (!isVisited[x][y] && grid[x][y] == 0)
    				grid[x][y] = 2;
    		}
    	}
    }
    
    public List<String> generateAbbreviations(String word) {
    	List<String> list = new ArrayList<> ();
    	int length = word.length();
    	for (int mask = 0; mask < (1<<length); mask++){
    		StringBuilder sb = new StringBuilder();
    		int count = 0;
    		for (int i = length - 1; i >= 0; i--) {
    			if ((mask & (1 << i)) == 0) {
    				if (count != 0) {
    					sb.append(count);
    					count = 0;
    				}
    				sb.append(word.charAt(length - i - 1));
    			}
    			else {
    				count++;
    			}
    		}
    		// be careful, if count is not zero, we still need to append it
    		if (count != 0)
    			sb.append(count);
    		list.add(sb.toString());
    	}
    	return list;
    	
    }
    
    String[] sMap = new String[26];
    HashMap<String, Character> cMap = new HashMap<>();
    public boolean wordPatternMatch(String pattern, String str) {
        return dfs (pattern, 0, str, 0);
    }
    
    public boolean dfs (String pattern, int pIdx, String str, int sIdx) {
    	if (sIdx >= str.length() || pIdx >= pattern.length())
    		return pIdx == pattern.length() && sIdx == str.length();
    	char c = pattern.charAt(pIdx);
    	if (sMap[c - 'a'] == null) {
    		for (int j = sIdx + 1; j <= str.length(); j++) {
    			String substring = str.substring(sIdx, j);
    			if (cMap.containsKey(substring) && cMap.get(substring) != null)
    				continue;
    			sMap[c - 'a'] = substring;
    			cMap.put(substring, c);
    			if (dfs (pattern, pIdx + 1, str, j))
    				return true;
    			cMap.put(substring, null);
    		}
    		sMap[c - 'a'] = null;
    		return false;
    	}
    	else {
    		String patch = sMap[c - 'a'];
    		if (sIdx + patch.length() > str.length())
    			return false;
    		if (!patch.equals(str.substring(sIdx, sIdx + patch.length())))
    			return false;
    		return dfs(pattern, pIdx + 1, str, sIdx + patch.length());
    	}
    }
    
    
    
    public void dfs(int[] board, int m, int n, int current, int count) {
    	count++;
    	board[current] = 1;
    	//System.out.println(Arrays.toString(board));
    	if (count == n) {
    		maxCount++;
    		board[current] = 0;
    		return;
    	}
    	if (count >= m)
    		maxCount++;
    	for (int i = 0; i < 9; i++) {
    		if (isNeighbor(board, current, i)) {
    			dfs(board, m, n, i, count);
    		}
    	}
    	board[current] = 0;
    }
    
    public boolean isNeighbor(int[] board, int current, int neighbor) {
    	if (board[neighbor] == 1 || current == neighbor)
    		return false;
    	int x = current / 3;
    	int y = current % 3;
    	int nx = neighbor / 3;
    	int ny = neighbor % 3;
    	if (Math.abs(nx - x) <= 1 && Math.abs(ny - y) <= 1)
    		return true;
    	if (Math.abs(nx - x) == 2 && Math.abs(ny - y) == 0 ) {
    		return board[3 + ny] == 1; 
    	}
    	if (Math.abs(ny - y) == 2 && Math.abs(nx - x) == 0) {
    		return board[nx * 3 + 1] == 1; 
    	}
    	if (Math.abs(nx - x) == 2 && Math.abs(ny - y) == 1) {
    		return true;
    	}
    	if (Math.abs(ny - y) == 2 && Math.abs(nx - x) == 1) {
    		return true;
    	}
//    	if (Math.abs(nx - x) == 2 && Math.abs(ny - y) <= 1) {
//    		return board[3 + ny] == 1 && board[3 + y] == 1; 
//    	}
//    	if (Math.abs(ny - y) == 2 && Math.abs(nx - x) <= 1) {
//    		return board[nx * 3 + 1] == 1 && board[x * 3 + 1] == 1; 
//    	}
    	return board[4] == 1;
    }
	
	public List<String> wordBreak(String s, Set<String> wordDict) {
        map = new HashMap<>();
        return wordBreakHelper(s, wordDict);
    }
	
	HashMap<String, List<String>> map;
	private List<String> wordBreakHelper(String s, Set<String> wordDict) {
		if (map.containsKey(s))
			return map.get(s);
		List<String> ret = new ArrayList<>();
		if (wordDict.contains(s))
			ret.add(s);
		for (int i = 1; i<s.length(); i++) {
			String prefix = s.substring(0, i);
			if (wordDict.contains(prefix)) {
				for (String sufix: wordBreakHelper(s.substring(i), wordDict)) {
					ret.add(prefix + " " + sufix);
				}
			}
		}
		map.put(s, ret);
		return ret;
	}
	
	public boolean solve (char[][] board, int x, int y) {
		if (y>=9)
			return solve(board, x+1, 0);
		if (x>=9)
			return true;
		if (board[x][y] != '.')
			return solve(board, x, y+1);
		for (int feasible = 1; feasible <=9; feasible ++){
			char fill = (char)(feasible+'0');
			if (isValid(board, x, y, fill)) {
				board[x][y] = fill;
				if (solve (board,x, y+1))
					return true;
				else
					board[x][y] = '.';
			}
		}
		
		return false;
	}
	
	private boolean isValid(char[][] board, int x, int y, char n) {
		for (int i = 0; i<9; i++) {
			if (board[i][y] == n)
				return false;
		}
		for (int j = 0; j < 9; j++) {
			if (board[x][j] == n)
				return false;
		}
		for (int i = x/3*3; i< x/3*3+3; i++) {
			for (int j = y/3*3; j< y/3*3+3; j++){
				if (board[i][j] == n)
					return false;
			}
		}
		return true;
	}
	
	private void permuteUnique(int[] nums, HashMap<Integer, Integer> isVisited,
			List<List<Integer>> result, LinkedList<Integer> list) {
		if (list.size()==nums.length) {
			ArrayList<Integer> al = new ArrayList<>();
			al.addAll(list);
			result.add(al);
			return;
		}
		for (int element: isVisited.keySet()){
			int freq = isVisited.get(element);
			if(freq>0){
				isVisited.put(element, freq-1);
				list.add(element);
				permuteUnique(nums, isVisited, result, list);
				isVisited.put(element, freq);
				list.removeLast();
			}
		}
	}
	
    public List<String> findWords(char[][] board, String[] words) {
    	List<String> list = new ArrayList<>();
        if (board==null||board.length==0||words==null||words.length==0)
        	return list;
        for (String word:words) {
        	outerloop:
	        for (int i = 0; i<board.length; i++){
	        	for (int j = 0; j<board[0].length; j++){
	        		if (findWordsDFS(board, word, 0, i, j)) {
	        			list.add(word);
	        			break outerloop;
	        		}
	        	}
	        }
        }
        return list;
    }
    
    public boolean findWordsDFS(char[][] board, String word, int pos, int i, int j){
    	if (pos>=word.length())
    		return true;
    	if (board[i][j]!=word.charAt(pos))
    		return false;
    	int[] dx = new int[]{-1, 1, 0, 0};
    	int[] dy = new int[]{0, 0, 1, 1};
    	board[i][j] = '#';
    	for (int index=0; index<4; index++){
    		int nx = i+dx[index];
    		int ny = j+dy[index];
    		if (nx>=0 && nx<board.length && ny>=0 && ny<board[0].length){
    			if (findWordsDFS(board, word, pos+1, nx, ny)) {
    				board[i][j] = word.charAt(pos);
    				return true;
    			}
    		}
    	}
    	board[i][j] = word.charAt(pos);
    	return false;   	
    }
    
    int[] dx = new int[] {-1, 1, 0, 0};
    int[] dy = new int[] {0, 0, 1, 1};
    
    public boolean exist(char[][] board, String word) {
    	if (board == null || board.length == 0 || word == null || word.length() == 0)
    		return false;
    	for (int i = 0; i < board.length; i++) {
    		for (int j = 0; j < board[0].length; j++) {
    			if (dfs(board, word, 0, i, j))
    				return true;
    		}
    	}
    	return false;
    }
    
    public List<List<String>> solveNQueens(int n) {
    	List<List<String>> boards = new ArrayList<> ();
    	if (n <= 0)
    		return boards;
    	char[][] board = new char[n][n];
    	for (int i = 0; i < n; i++) {
    		for (int j = 0; j < n; j++) {
    			board[i][j] = '.';
    		}
    	}
    	boolean[] cols = new boolean[n];
    	solveNQueensHelper(board, 0, cols, boards);
    	return boards;
    }
    
    public void solveNQueensHelper (char[][] board, int row, boolean[] cols,
    		List<List<String>> result) {
    	if (row == board.length) {
    		List<String> newBoard = new ArrayList<> ();
    		for (char[] a_row : board) {
    			newBoard.add(String.copyValueOf(a_row));
    		}
    		result.add(newBoard);
    		return;
    	}
    	int n = board.length;
    	for (int col = 0; col < n; col++) {
    		boolean diagCertify = true;
    		if (!cols[col]) {
    			//search the diagonal
    			for (int offset = 1; 
    					offset <= Math.min(row, Math.max(col,  n - col - 1)); offset++) {
    				
    				if (col - offset >= 0 && board[row - offset][col - offset] == 'Q') {
    					diagCertify = false;
    					break;
    				}
    				if (col + offset < n && board[row - offset][col + offset] == 'Q') {
    					diagCertify = false;
    					break;
    				}
    			}
    			if (diagCertify) {
    				board[row][col] = 'Q';
    				cols[col] = true;
    				solveNQueensHelper(board, row + 1, cols, result);
    				board[row][col] = '.';
    				cols[col] = false;
    			}
    		}
    	}
    }
    
    private boolean dfs(char[][] board, String word, int pos, int i, int j) {
    	if (word.charAt(pos) != board[i][j])
    		return false;
    	if (pos == word.length()-1)
    		return true;
    	char temp = board[i][j];
    	board[i][j] = '#';
    	for (int idx = 0; idx<4; idx++) {
    		int nx = i + dx[idx];
    		int ny = j + dy[idx];
    		if (nx >= 0 && nx < board.length && ny >= 0 && ny < board[0].length) {
    			if (dfs(board, word, pos+1, nx, ny))
    				return true;
    		}
    	}
    	board[i][j] = temp;
    	return false;
    }
    
    Set<Integer> set = new HashSet<>();
    public List<Integer> grayCode(int n) {
        LinkedList<Integer> list = new LinkedList<>();
        dfs (list, 0, 1<<n, n);
        return list;
    }
    
    public int totalNQueens(int n) {
    	return dfs(new int[n], 0);
    }
    
    public int dfs(int[] board, int row) {
    	int count = 0;
    	if (row == board.length)
    		return 1;
    	for (int i = 0; i < board.length; i++) {
    		if (isValid(board, row, i)) {
    			board[row] = i;
    			count += dfs(board, row + 1);
    		}
    	}
    	return count;
    }
    
    public boolean isValid(int[] board, int row, int col) {
    	for (int i = 0; i < row; i++) {
    		if (board[i] == col || Math.abs(board[i] - col) == row - i)
    			return false;
    	}
    	return true;
    }
    
    public void dfs (LinkedList<Integer> ret, int current, int size, int n) {
    	ret.add(current);
    	set.add(current);
    	if (ret.size() == size)
    		return;
    	for (int i = 0; i<n; i++) {
    		int neighbor = getNeighbor(current, i);
    		if (!set.contains(neighbor)) {
    			dfs (ret, neighbor, size, n);
    			if (ret.size() == size)
    				return;
    		}
    	}
    	ret.removeLast();
    	set.remove(current);
    }
    
    public int getNeighbor (int self, int digit){
    	int mask = 1<<digit;
    	return self^mask;
    }
    
	public static void main (String[] args) {
		Solutions s = new Solutions();
		System.out.println(s.wordPatternMatch("bbbb", "asdasdasdasd"));
				
	}

}
