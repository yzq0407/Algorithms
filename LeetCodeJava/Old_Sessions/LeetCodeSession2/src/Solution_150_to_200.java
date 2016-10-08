import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;

public class Solution_150_to_200 {
	public class TwoSum {
	    HashMap<Integer, Integer> map = new HashMap<>();

	    // Add the number to an internal data structure.
		public void add(int number) {
		    int freq = map.containsKey(number)? map.get(number): 0;
		    map.put(number, freq + 1);
		}

	    // Find if there exists any pair of numbers which sum is equal to the value.
		public boolean find(int value) {
		    for (Integer key: map.keySet()) {
		        int rem = value - key;
		        if (rem != key && map.containsKey(rem)) {
		            return true;
		        }
		        else if (rem == key && map.get(key) >= 2){
		            return true;
		        }
		    }
		    return false;
		}
	}
	
	public int compareVersion(String version1, String version2) {
		version1.replace(".", " ");
		version2.replace(".", " ");
        String[] v1 = version1.split(" ");
        String[] v2 = version2.split(" ");
        System.out.println(Arrays.toString(v1));
        if (compareComponent(v1[0], v2[0]) != 0) {
            return compareComponent(v1[0], v2[0]);
        }
        return compareComponent(v1[1], v2[1]);
    }
	
	public int calculateMinimumHP(int[][] dungeon) {
        int n = dungeon.length;
        int m = dungeon[0].length;
        int[][] dp = new int[n][m];
        dp[n - 1][m - 1] = dungeon[n - 1][m - 1] >= 0? 1: -dungeon[n - 1][m - 1] + 1;
        for (int j = m - 2; j >= 0; j--) {
            dp[n - 1][j] = dp[n - 1][j + 1] - dungeon[n - 1][j];
            if (dp[n - 1][j] <= 0) {
                dp[n - 1][j] = 1;
            }
        }
        for (int i = n - 2; i >= 0; --i) {
            dp[i][m -1] = dp[i + 1][m - 1] - dungeon[i][m - 1];
            if (dp[i][m - 1] <= 0) {
                dp[i][m - 1] = 1;
            }
        }
        for(int i = n - 2; i >= 0; i--) {
            for (int j = m - 2; j >= 0; --j) {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j];
                if (dp[i][j] <= 0) {
                    dp[i][j] = 1;
                }
            }
        }
        return dp[0][0];
    }
    
    public int compareComponent(String v1, String v2) {
        int p1 = 0;
        int p2 = 0;
        while (v1.charAt(p1) =='0') {
            p1++;
        }
        while (v2.charAt(p2) =='0'){
            p2++;
        }
        if ((v1.length() - p1) !=(v2.length() - p2)) {
            return (v1.length() - p1) - (v2.length() - p2);
        }
        while (p1 < v1.length()) {
            if (v1.charAt(p1) != v2.charAt(p2)){
                return v1.charAt(p1) - v2.charAt(p2);
            }
            p1++;
            p2++;
        }
        return 0;
    }
	
	public String largestNumber(int[] nums) {
		String[] nums_str = new String[nums.length];
		for (int i = 0; i < nums.length; ++i) {
			nums_str[i] = String.valueOf(nums[i]);
		}
		Arrays.sort(nums_str, new Comparator<String>(){

			@Override
			public int compare(String o1, String o2) {
				// TODO Auto-generated method stub
				return (o1 + o2).compareTo(o2 + o1);
			}
		});
		StringBuilder sb = new StringBuilder();
		for(int i = nums.length - 1; i != 0; i--) {
			sb.append(nums_str[i]);
		}
		return sb.toString();
    }
	
	public int reverseBits(int n) {
        int rv = 0;
        for (int i = 31; i>=0; i--) {
            int mask = 1 << i;
            if ((mask & n) != 0) {
                rv |= (1 << (31 - i));
            }
        }
        return rv;
    }
	
	public int maximumGap(int[] nums) {
        if (nums.length < 2) {
            return 0;
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int num: nums) {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
        int gap = (max - min) / (nums.length - 1);
        int[] mins = new int[nums.length];
        int[] maxs = new int[nums.length];
        Arrays.fill(mins, Integer.MAX_VALUE);
        Arrays.fill(maxs, Integer.MIN_VALUE);
        for (int num : nums) {
            int key = (num - min) / gap;
            mins[key] = Math.min(mins[key], num);
            maxs[key] = Math.max(maxs[key], num);
        }
        System.out.println(Arrays.toString(mins));
        System.out.println(Arrays.toString(maxs));
        int maxgap = 0;
        int last_max = maxs[0];
        for (int key = 1; key < nums.length; key++) {
            if (mins[key] != Integer.MAX_VALUE){
            	maxgap = Math.max(mins[key] - last_max, maxgap);
            	last_max = maxs[key];
            }
            
        }
        return maxgap;
    }
	
	public String convertToTitle(int n) {
        StringBuilder sb = new StringBuilder();
        while (n != 0) {
            sb.insert(0, (char)((n - 1) % 26 + 'A'));
            n = (n - 1) / 26;
        }
        return sb.toString();
    }
	
	public String reverseWords(String s) {
        char[] array = s.toCharArray();
        int i = 0;
        int j = array.length - 1;
        while (i < j) {
            char temp = array[i];
            array[i++] = array[j];
            array[j--] = temp;
        }
        i = 0;
        while (i < array.length) {
            j = i;
            while (j != array.length && array[j] != ' ') {
                j++;
            }
            int ii = i;
            int jj = j - 1;
            while (ii < jj) {
                char temp = array[ii];
                array[ii++] = array[jj];
                array[jj--] = temp;
            }
            i = j + 1;
        }
        return String.valueOf(array);
    }
	
	public int lengthOfLongestSubstringTwoDistinct(String s) {
        int[] count = new int[128];
        int i = 0, j = 0;
        int max = 0;
        int countDistinct = 0;
        while (i < s.length() && j < s.length()) {
            while (j < s.length() && countDistinct < 3) {
                if (count[s.charAt(j++)]++ == 0)
                    countDistinct++;
            }
            max = Math.max(countDistinct == 3? j - i - 1: j - i, max);
            while (countDistinct > 2 && i < j) {
                if (--count[s.charAt(i++)] == 0)
                    countDistinct--;
            }
        }
        return max;
    }
	
	public String fractionToDecimal(int numerator, int denominator) {
        int integer = numerator / denominator;
        long rem =  numerator % denominator;
        if (numerator == 0) return String.valueOf(integer);
        StringBuilder sb = new StringBuilder();
        //rem --> idx of rem
        HashMap<Long, Integer> map = new HashMap<>();
        map.put(rem, 0);
        for (int idx = 1; rem != 0; idx++) {
            //int decimal = rem * 10 / denominator;
            sb.append(rem * 10 / denominator);
            rem = rem * 10 % denominator;
            if (map.containsKey(rem)) {
                sb.insert(map.get(rem), "(");
                sb.append(")");
                break;
            }
            else {
                map.put(rem, idx);
            }
        }
        return String.valueOf(integer) + "." + sb.toString();
    }
	
	public static void main(String[] args){
		Solution_150_to_200 s= new Solution_150_to_200();
		//int[][] test = new int[][]{{-2, -3, 3}, {-5, -10, 1}, {10, 30, -5}};
		int[] test = new int[]{5, 2, 100, 74, 23, 24};
		System.out.println(s.fractionToDecimal(1, 5));
	}

}
