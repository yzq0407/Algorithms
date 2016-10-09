import java.util.*;


public class Contest8_Oct_8 {

    public String addStrings(String num1, String num2) {
        if (num1.length() < num2.length())  return addStrings(num2, num1);
        StringBuilder sb = new StringBuilder();
        int pos = 1, carry = 0;
        for (pos = 1; pos <= num2.length(); pos++) {
            int c1 = num1.charAt(num1.length() - pos) - '0';
            int c2 = num2.charAt(num2.length() - pos) - '0';
            sb.append('0' + (c1 + c2 + carry) % 10);
            carry = (c1 + c2 + carry) / 10;
        }
        while (pos <= num1.length()){
            int c = num1.charAt(num1.length() - pos) - '0';
            sb.append('0' + (c + carry) % 10);
            carry = (c + carry) / 10;
        }
        if (carry != 0) sb.append('1');
        return sb.reverse().toString();

    }

    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num: nums) {
            sum += num;
        }
        if (sum %2 == 1)    return false;
        int target = sum / 2;
        boolean[][] dp = new boolean[target + 1][nums.length + 1];
        dp[0][0] = true;
        for (int i = 0; i < nums.length; ++i) {
            for (int s = 0; s <= target; ++s) {
                dp[s][i + 1] = dp[s][i];
                if (s - nums[i] >=0 && dp[s - nums[i]][i])
                    dp[s][i + 1] = true;
            }
        }
            
        return dp[target][nums.length];
    }

    int[] dx = new int[]{-1, 1, 0, 0};
    int[] dy = new int[]{0, 0, -1, 1};
    public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> res = new ArrayList<>();
        if (matrix.length == 0) return res;
        int n = matrix.length, m = matrix[0].length;
        //pacific
        int[][] count = new int[n][m];
        boolean[][] visited = new boolean[n][m];
        for (int i = 0; i < n; ++i) {
            if (!visited[i][0])
                dfs(count, matrix, visited, i, 0);
        }
        for (int j = 0; j < m; ++j) {
            if (!visited[0][j]) dfs(count, matrix, visited, 0, j);
        }
        visited = new boolean[n][m];
        for (int i = 0; i < n; ++i) {
            if (!visited[i][m - 1])  dfs(count, matrix, visited, i, m - 1);
        }
        for (int j = 0; j < m; ++j) {
            if (!visited[n - 1][j]) dfs(count, matrix, visited, n - 1, j);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (count[i][j] == 2)   res.add(new int[]{i, j});
            }
        }
        return res;
    }

    public void dfs(int[][] count, int[][] matrix, boolean[][] visited, int i, int j) {
        ++count[i][j];
        visited[i][j] = true;
        for (int d = 0; d < 4; ++d) {
            int nx = dx[d] + i, ny = dy[d] + j;
            if (nx < count.length && nx >= 0 && ny >= 0 && 
                    ny < count[0].length && !visited[nx][ny] && matrix[nx][ny] >= matrix[i][j]) {
                dfs(count, matrix, visited, nx, ny);
                    }
        }
    }

    public int wordsTyping(String[] sentence, int rows, int cols) {
        int times = 0, pos = 0, col = 0, idx = 0;
        while (pos < rows) {
            if (idx == sentence.length) {
                idx = 0;
                ++times;
                continue;
            }
            if (sentence[idx].length() > cols)  return 0;
            if (cols - col >= sentence[idx].length()) {
                col += (sentence[idx++].length() + 1);
            }
            else {
                ++pos;
                col = 0;
            }
        }
        return times;
    }


    public static void main(String[] args) {
        Contest8_Oct_8 solution = new Contest8_Oct_8();
	int[] nums = new int[] {66,90,7,6,32,16,2,78,69,88,85,26,3,9,58,65,30,96,11,31,99,49,63,83,79,97,20,64,81,80,25,69,9,75,23,70,26,71,25,54,1,40,41,82,32,10,26,33,50,71,5,91,59,96,9,15,46,70,26,32,49,35,80,21,34,95,51,66,17,71,28,88,46,21,31,71,42,2,98,96,40,65,92,43,68,14,98,38,13,77,14,13,60,79,52,46,9,13,25,8};
        System.out.println(solution.canPartition(nums));

    }
}
