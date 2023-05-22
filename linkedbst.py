"""
File: linkedbst.py
Author: Ken Lambert
"""
from abstractcollection import AbstractCollection
from bstnode import BSTNode
from linkedstack import LinkedStack
from linkedqueue import LinkedQueue
from math import log

from matplotlib import pyplot as plt
import random
import time
import random
import copy


class LinkedBST(AbstractCollection):
    """An link-based binary search tree implementation."""

    def __init__(self, sourceCollection=None):
        """Sets the initial state of self, which includes the
        contents of sourceCollection, if it's present."""
        self._root = None
        AbstractCollection.__init__(self, sourceCollection)

    # Accessor methods
    def __str__(self):
        """Returns a string representation with the tree rotated
        90 degrees counterclockwise."""

        def recurse(node, level):
            s = ""
            if node != None:
                s += recurse(node.right, level + 1)
                s += "| " * level
                s += str(node.data) + "\n"
                s += recurse(node.left, level + 1)
            return s

        return recurse(self._root, 0)

    def __iter__(self):
        """Supports a preorder traversal on a view of self."""
        if not self.isEmpty():
            stack = LinkedStack()
            stack.push(self._root)
            while not stack.isEmpty():
                node = stack.pop()
                yield node.data
                if node.right != None:
                    stack.push(node.right)
                if node.left != None:
                    stack.push(node.left)

    def preorder(self):
        """Supports a preorder traversal on a view of self."""
        return None

    def inorder(self):
        """Supports an inorder traversal on a view of self."""
        lyst = list()

        def recurse(node):
            if node != None:
                recurse(node.left)
                lyst.append(node.data)
                recurse(node.right)

        recurse(self._root)
        return iter(lyst)

    def inorder_nodes(self):
        """Supports an inorder traversal on a view of self."""
        lyst = list()

        def recurse(node):
            if node != None:
                recurse(node.left)
                lyst.append(node)
                recurse(node.right)

        recurse(self._root)
        return iter(lyst)

    def postorder(self):
        """Supports a postorder traversal on a view of self."""
        return None

    def levelorder(self):
        """Supports a levelorder traversal on a view of self."""
        return None

    def __contains__(self, item):
        """Returns True if target is found or False otherwise."""
        return self.find(item) != None

    def find(self, item):
        """If item matches an item in self, returns the
        matched item, or None otherwise."""

        def recurse(node):
            if node is None:
                return None
            elif item == node.data:
                return node.data
            elif item < node.data:
                return recurse(node.left)
            else:
                return recurse(node.right)

        return recurse(self._root)

    def find_while(self, item):
        """If item matches an item in self, returns the
        matched item, or None otherwise. Using while."""
        if self.isEmpty():
            return None
        else:
            root = self._root
            while True:
                if item == root.data:
                    return root
                elif item < root.data:
                    if root.left is not None:
                        root = root.left
                    else:
                        return None
                else:
                    if root.right is not None:
                        root = root.right
                    else:
                        return None

    # Mutator methods
    def clear(self):
        """Makes self become empty."""
        self._root = None
        self._size = 0

    def add(self, item):
        """Adds item to the tree."""

        # Helper function to search for item's position
        def recurse(node):
            # New item is less, go left until spot is found
            if item < node.data:
                if node.left == None:
                    node.left = BSTNode(item)
                else:
                    recurse(node.left)
            # New item is greater or equal,
            # go right until spot is found
            elif node.right == None:
                node.right = BSTNode(item)
            else:
                recurse(node.right)
                # End of recurse

        # Tree is empty, so new item goes at the root
        if self.isEmpty():
            self._root = BSTNode(item)
        # Otherwise, search for the item's spot
        else:
            recurse(self._root)
        self._size += 1

    def add_while(self, item):
        """Adds item to the tree using while."""
        if self.isEmpty():
            self._root = BSTNode(item)
        else:
            root = self._root
            while True:
                if item < root.data:
                    if root.left is not None:
                        root=root.left
                    else:
                        root.left = BSTNode(item)
                        break
                else:
                    if root.right is not None:
                        root=root.right
                    else:
                        root.right = BSTNode(item)
                        break
        self._size += 1

    def remove(self, item):
        """Precondition: item is in self.
        Raises: KeyError if item is not in self.
        postcondition: item is removed from self."""
        if not item in self:
            raise KeyError("Item not in tree.""")

        # Helper function to adjust placement of an item
        def liftMaxInLeftSubtreeToTop(top):
            # Replace top's datum with the maximum datum in the left subtree
            # Pre:  top has a left child
            # Post: the maximum node in top's left subtree
            #       has been removed
            # Post: top.data = maximum value in top's left subtree
            parent = top
            currentNode = top.left
            while not currentNode.right == None:
                parent = currentNode
                currentNode = currentNode.right
            top.data = currentNode.data
            if parent == top:
                top.left = currentNode.left
            else:
                parent.right = currentNode.left

        # Begin main part of the method
        if self.isEmpty(): return None

        # Attempt to locate the node containing the item
        itemRemoved = None
        preRoot = BSTNode(None)
        preRoot.left = self._root
        parent = preRoot
        direction = 'L'
        currentNode = self._root
        while not currentNode == None:
            if currentNode.data == item:
                itemRemoved = currentNode.data
                break
            parent = currentNode
            if currentNode.data > item:
                direction = 'L'
                currentNode = currentNode.left
            else:
                direction = 'R'
                currentNode = currentNode.right

        # Return None if the item is absent
        if itemRemoved == None: return None

        # The item is present, so remove its node

        # Case 1: The node has a left and a right child
        #         Replace the node's value with the maximum value in the
        #         left subtree
        #         Delete the maximium node in the left subtree
        if not currentNode.left == None \
                and not currentNode.right == None:
            liftMaxInLeftSubtreeToTop(currentNode)
        else:

            # Case 2: The node has no left child
            if currentNode.left == None:
                newChild = currentNode.right

                # Case 3: The node has no right child
            else:
                newChild = currentNode.left

                # Case 2 & 3: Tie the parent to the new child
            if direction == 'L':
                parent.left = newChild
            else:
                parent.right = newChild

        # All cases: Reset the root (if it hasn't changed no harm done)
        #            Decrement the collection's size counter
        #            Return the item
        self._size -= 1
        if self.isEmpty():
            self._root = None
        else:
            self._root = preRoot.left
        return itemRemoved

    def replace(self, item, newItem):
        """
        If item is in self, replaces it with newItem and
        returns the old item, or returns None otherwise."""
        probe = self._root
        while probe != None:
            if probe.data == item:
                oldData = probe.data
                probe.data = newItem
                return oldData
            elif probe.data > item:
                probe = probe.left
            else:
                probe = probe.right
        return None

    def is_leaf(self, p):
        '''
        Documentation.
        '''
        return p.left is None and p.right is None

    def parent(self, p):
        '''
        Documentation.
        '''
        for _p in self.inorder_nodes():
            if _p.left is p or _p.right is p:
                return _p

    def depth(self, p):
        '''
        Documentation.
        '''
        if p is self._root:
            return 0
        else:
            return 1 + self.depth(self.parent(p))

    def height(self):
        '''
        Return the height of tree
        :return: int
        '''
        if self._root:
            return max(self.depth(p) for p in self.inorder_nodes() if self.is_leaf(p))
        else:
            return -1

    def is_balanced(self):
        '''
        Return True if tree is balanced
        :return:
        '''
        return self.height() < 2 * log(self._size + 1, 2) - 1

    def range_find(self, low, high):
        '''
        Returns a list of the items in the tree, where low <= item <= high."""
        :param low:
        :param high:
        :return:
        '''
        res = []
        for _p in self.inorder_nodes():
            if low <= _p.data <= high:
                res.append(_p.data)
        return res

    def rebalance(self):
        '''
        Rebalances the tree.
        :return:
        '''
        sym_or = []
        for _p in self.inorder():
            sym_or.append(_p)
        self.clear()
        def fill_tree(nodes):
            if nodes:
                length = len(nodes)
                self.add(nodes[length//2])
                fill_tree(nodes[:length//2])
                fill_tree(nodes[length//2 + 1:])
            else:
                return

        fill_tree(sym_or)


    def successor(self, item):
        """
        Returns the smallest item that is larger than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        greater = []
        for _p in self.inorder_nodes():
            if _p.data > item:
                greater.append(_p.data)
        if greater:
            return min(greater)
        else:
            return

    def predecessor(self, item):
        """
        Returns the largest item that is smaller than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        lower = []
        for _p in self.inorder_nodes():
            if _p.data < item:
                lower.append(_p.data)
        if lower:
            return max(lower)
        else:
            return

    def demo_bst(self, path):
        """
        Demonstration of efficiency binary search tree for the search tasks.
        :param path:
        :type path:
        :return:
        :rtype:
        """
        with open(path,'r',encoding='utf-8') as _f:
            num_words = 10000
            sorted_content = sorted(_f.readlines())
            time1 = 0
            time2 = 0
            time3 = 0
            time4 = 0
            w_sample = random.sample(sorted_content, num_words) #10000 random words

            randomized_content = copy.deepcopy(sorted_content)
            random.shuffle(randomized_content)

            tree1 = LinkedBST()
            tree2 = LinkedBST()

            for item in sorted_content:
                tree1.add_while(item)
            print('Sorted BST filled...')
            for item in randomized_content:
                tree2.add_while(item)
            print('Shuffled BST filled...')
            for word in w_sample:
                _t1 = time.time()
                _var = sorted_content.index(word)
                _t2 = time.time()
                time1 += _t2 - _t1

                _t1 = time.time()
                _var = tree1.find_while(word)
                _t2 = time.time()
                time2 += _t2 - _t1

                _t1 = time.time()
                _var = tree2.find_while(word)
                _t2 = time.time()
                time3 += _t2 - _t1

            tree2.rebalance()

            for word in w_sample:
                _t1 = time.time()
                _var = tree2.find_while(word)
                _t2 = time.time()
                time4 += _t2 - _t1
            print('Time checked...')
            print('Done!')
            values = [time1, time2, time3, time4]
            labels = ['index()', 'sorted BST', 'shuffled BST', 'balanced BST']
            with open('results.txt', 'w', encoding='utf-8') as res_f:
                res_str = ''
                for _i in range(4):
                    res_str = res_str + labels[_i] + ': ' + str(values[_i]) + '\n'
                res_f.write(res_str)
            plt.bar(labels, values)
            plt.xlabel('function')
            plt.ylabel('time, s')
            plt.show()

test = LinkedBST()
test.demo_bst('words.txt')