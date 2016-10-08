
public class NumArray {
//	int[] fTree;
//	int[] vals;
	
//	public NumArray(int[] nums) {
//		vals = new int[nums.length];
//        fTree = new int[nums.length+1];
//        for (int i = 0; i<nums.length; i++) {
//        	update(i, nums[i]);
//        }
//    }
//
//    void update(int i, int val) {
//    	i = i+1;
//    	int diff = val - vals[i-1];
//    	vals[i-1] = val;
//        while (i<fTree.length) {
//        	fTree[i] = fTree[i] + diff;
//        	i += (i&-i);
//        }
//    }
//    
//    private int search (int i) {
//    	int ret = 0;
//    	while (i>0) {
//    		ret += fTree[i];
//    		i -= (i&-i);
//    	}
//    	return ret;
//    }
//
//    public int sumRange(int i, int j) {
//        return search(j+1) - search(i);
//    }

	
	// the easy version
	
	int[] runningSum;
    public NumArray(int[] nums) {
    	runningSum = new int[nums.length+1];
    	for (int i = 0; i<nums.length; i++){
    		runningSum[i+1] = runningSum[i] + nums[i];
    	}
    }

    public int sumRange(int i, int j) {
        return runningSum[j+1] - runningSum[i];
    }
	public static void main(String[] args) {
		int[] arr = new int[] {1, 5, 7, 8, 9, 10};
		NumArray na = new NumArray(arr);
//		na.update(3, -1);
//		na.update(5, 1);
//		na.update(5, 4);
		System.out.println(na.sumRange(1, 5));
	}
}
