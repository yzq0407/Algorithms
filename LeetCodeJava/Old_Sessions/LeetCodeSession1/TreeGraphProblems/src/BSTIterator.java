import java.util.Stack;

public class BSTIterator {
	Stack<TreeNode> stack;

    public BSTIterator(TreeNode root) {
    	stack = new Stack<> ();
    	while (root != null) {
    		stack.push(root);
    		root = root.left;
    	}
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    /** @return the next smallest number */
    public int next() {
        TreeNode ret = stack.pop();
        int val = ret.val;
        if (ret.right != null) {
        	ret = ret.right;
        	stack.push(ret);
        	while (ret.left != null) {
        		ret = ret.left;
        		stack.push(ret);
        	}
        }
        return val;
    }
}
