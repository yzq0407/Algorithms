import java.util.ArrayList;
import java.util.List;

public class PalindromePairs_336 {
	class TrieNode {
	    TrieNode[] children;
	    int index;
	    // Initialize your data structure here.
	    public TrieNode() {
	        children = new TrieNode[26];
	        index = -1;
	    }
	}
	TrieNode root = new TrieNode();
	public List<List<Integer>> palindromePairs(String[] words) {
		List<List<Integer>> rv = new ArrayList<>();
		if (words == null || words.length == 0) {
			return rv;
		}
        root = new TrieNode();
        for (int index = 0; index < words.length; ++index) {
        	insert(words[index], index);
        }
        for (int index = 0; index < words.length; ++index) {
        	//for a word there are several cases that it can case
        	//it's
        	char[] cs = words[index].toCharArray();
        	int rc = search(words[index], words[index].length() - 1, 0);
        	//if rc == index, this means rc itself is a palindrome, it can concatenate 
        	//with any empty string
        	if (rc != -1) {
        		if (rc == index && root.index != -1 && root.index != rc) {
        			ArrayList<Integer> list = new ArrayList<>();
        			list.add(index);
        			list.add(root.index);
        			rv.add(list);
        			ArrayList<Integer> list2 = new ArrayList<>();
        			list2.add(root.index);
        			list2.add(index);
        			rv.add(list2);
        		}
        		//if rc != index, this means rc has another reverse string, we add it into 
        		//the rv
        		if (rc != index) {
        			ArrayList<Integer> list = new ArrayList<>();
        			list.add(index);
        			list.add(rc);
        			rv.add(list);
        		}
        	}
        	//if 0 -- j is a palindrome
        	for (int i = 0; i < words[index].length(); ++i) {
        		if (i != words[index].length() - 1 && isPalindrome(cs, 0, i)) {
        			rc = search(words[index], words[index].length() - 1, i + 1);
        			if (rc != -1) {
        				ArrayList<Integer> list = new ArrayList<>();
            			list.add(rc);
            			list.add(index);
            			rv.add(list);
        			}
        		}
        		if (i != 0 && isPalindrome(cs, i, words[index].length() - 1)) {
        			rc = search(words[index], i - 1, 0);
        			if (rc != -1) {
        				ArrayList<Integer> list = new ArrayList<>();
            			list.add(index);
            			list.add(rc);
            			rv.add(list);
        			}
        		}
        	}
        }
        
        return rv;
        
    }
	
	public boolean isPalindrome(char[] cs, int from, int to) {
		while (from < to) {
			if (cs[from++] != cs[to--]) {
				return false;
			}
		}
		return true;
	}
	
	public void insert(String word, int index) {
        if (word.length() == 0){
            root.index = index;
            return;
        }
        root.children[word.charAt(0) - 'a'] = insert(root.children[word.charAt(0) - 'a'], word, 0, index);
    }
    
    private TrieNode insert(TrieNode node, String word, int pos, int index) {
        if (node == null) {
            node = new TrieNode();
        }
        if (pos == word.length() - 1){
            node.index = index;
        }
        else {
            node.children[word.charAt(pos + 1) - 'a'] = insert(node.children[word.charAt(pos + 1) - 'a'], word, pos + 1, index);
        }
        return node;
    }

    // Returns if the word is in the trie.
    public int search(String word, int from, int to) {
        if (word.length() == 0) {
            return root.index;
        }
        return search(word, from, root.children[word.charAt(from) - 'a'], to);
    }
    
    private int search(String word, int pos, TrieNode node, int end) {
        if (node == null) {
            return -1;
        }
        if (pos == end) {
            return node.index;
        }
        int next = pos + (end - pos) / Math.abs(end - pos);
        return search(word, next, node.children[word.charAt(next) - 'a'], end);
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
    
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        
    }
    
    public static void main(String[] args) {
    	String[] test = new String[]{"bat", "tab", "cat", "battab", ""};
    	PalindromePairs_336 s = new PalindromePairs_336();
//    	for (int i = 0; i < test.length; ++i){
//    		s.insert(test[i], i);
//    	}
//    	System.out.println(s.search("sllsssll", 7, 3));
    	System.out.println(s.palindromePairs(test));
    }

}
