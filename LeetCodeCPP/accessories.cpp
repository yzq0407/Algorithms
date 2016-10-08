#include "accessories.h"

TreeNode::TreeNode(int v): val(v), left(nullptr), right(nullptr){}

ListNode::ListNode(int x): val(x), next(nullptr){}

Interval::Interval(int s, int e): start(s), end(e) {}

IndexedTreeNode::IndexedTreeNode(int v): val(v), smaller(0), duplicates(1), 
    left(nullptr), right(nullptr){}

