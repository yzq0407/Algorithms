#ifndef ALGO_ACCESSORIES
#define ALGO_ACCESSORIES
//a general binary treenode that has left child and right child
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int v);
};

//a general linkedlist node that has a next value
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x);
};

//a struct that represents a interval [start, end]
struct Interval {
    int start;
    int end;
    Interval(int s, int e);
};

//a binary search treenode that contains the number of nodes less than it 
//(in the left branch) and also number of duplicates
struct IndexedTreeNode {
    int val;
    int smaller;
    int duplicates;
    IndexedTreeNode* left;
    IndexedTreeNode* right;
    IndexedTreeNode(int v);
};
#endif
