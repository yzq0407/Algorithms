#ifndef ALGO_ACCESSORIES
#define ALGO_ACCESSORIES

#include <string>   //for trienode
#include <vector>   //for trienode
//a general binary treenode that has left child and right child
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    //ctor
    explicit TreeNode(int v): val(v), left(nullptr), right(nullptr){};

    //copy control
    TreeNode(const TreeNode& other) = delete;
    TreeNode operator=(const TreeNode& rhs) = delete;

    //dtor
    ~TreeNode() {
        delete(left);
        delete(right);
    }
};

//a general linkedlist node that has a next value
struct ListNode {
    int val;
    ListNode *next;

    //ctor
    explicit ListNode(int x): val(x), next(nullptr){};

    //copy control
    ListNode(const ListNode& other) = delete;
    ListNode operator=(const ListNode& rhs) = delete;

    //dtor
    ~ListNode() {
        delete next;
    }
};

//a struct that represents a interval [start, end]
struct Interval {
    int start;
    int end;

    //ctor
    explicit Interval(int s, int e): start(s), end(e){}
};

//a binary search treenode that contains the number of nodes less than it 
//(in the left branch) and also number of duplicates
struct IndexedTreeNode {
    int val;
    int smaller;
    int duplicates;
    IndexedTreeNode *left;
    IndexedTreeNode *right;
    
    //ctor
    explicit IndexedTreeNode(int v): val(v), smaller(0), duplicates(1),
    left(nullptr), right(nullptr){}

    //copy control
    IndexedTreeNode(const IndexedTreeNode& other) = delete;
    IndexedTreeNode operator=(const IndexedTreeNode& rhs) = delete;

    //dtor
    ~IndexedTreeNode() {
        delete left;
        delete right;
    }
};

//a trie node that contains lower case alphabetical characters, if the trienode 
//is a termination node, it will contains a non empty string pointer to the string value
struct TrieNode {
    //this will be nullptr if it is not a terminating node
    std::string val;
    std::vector<TrieNode*> children;
    
    //ctor
    explicit TrieNode (): val(""), children(26, nullptr) {}

    //copy control
    TrieNode(const TrieNode& other) = delete;
    TrieNode operator=(const TrieNode& rhs) = delete;

    //dtor
    ~TrieNode() {
        for (auto child: children) {
            delete(child);
        }
    }
};

//a different trienode that use unordered_map to store all the children
struct TrieNode_Map {
    std::string val;
    std::unordered_map<char, TrieNode_Map*> children;

    //ctor
    explicit TrieNode_Map(): val(""){}

    //copy control
    TrieNode_Map(const TrieNode_Map& other) = delete;
    TrieNode_Map operator=(const TrieNode_Map& rhs) = delete;

    //dtor
    ~TrieNode_Map() {
        for (auto pair: children) {
            delete(pair.second);
        }
    }
};

#endif
