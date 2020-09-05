# Swift-Algorithms

## 707. Design Linked List

### Use Single Linked List

使用哨兵节点当作伪head，这样的话就不需要讨论list为空的情况，而且可以把addAtHead和addAtTail都简化为addAtIndex的特殊形式。*链表的head是哨兵节点，真正的head可以用head.next访问*

```swift
class Node {
    let val: Int
    var next: Node?
    init(_ val: Int) {
        self.val = val
    } 
}

class MyLinkedList {
   
    private var head: Node?
    private var size: Int

    /** Initialize your data structure here. */
    init() {
        head = Node(0)
        size = 0
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
  // O(k), worst case O(n), where k is index
    func get(_ index: Int) -> Int {
        if index < 0 || index >= self.size {
            return -1
        }
        
        var curr = self.head
        for _ in stride(from: 0, to: index+1, by: 1) {
            curr = curr?.next
        }
        return (curr?.val)!
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
  // O(1)
    func addAtHead(_ val: Int) {
        self.addAtIndex(0, val)
    }
    
    /** Append a node of value val to the last element of the linked list. */
  // O(n)
    func addAtTail(_ val: Int) {
        self.addAtIndex(self.size, val)
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
  // O(k), worst case worst case O(n)
    func addAtIndex(_ index: Int, _ val: Int) {
        // if index is greater than linkedlist size
        // or index < 0, the node will not be inserted
        if index > self.size || index < 0 { return }
        
        self.size += 1
        var pred = self.head
        for _ in stride(from: 0, to: index, by: 1) {
            pred = pred?.next
        }
        let toAdd = Node(val)
        toAdd.next = pred?.next
        pred?.next = toAdd
        
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
  // O(k), worst case O(n)
    func deleteAtIndex(_ index: Int) {
        if index < 0 || index >= self.size { return }
        
        self.size -= 1
        var prev = self.head
        for _ in stride(from: 0, to: index, by: 1) {
            prev = prev?.next
        }
        prev?.next = prev?.next?.next
        
    }
}

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * let obj = MyLinkedList()
 * let ret_1: Int = obj.get(index)
 * obj.addAtHead(val)
 * obj.addAtTail(val)
 * obj.addAtIndex(index, val)
 * obj.deleteAtIndex(index)
 */
```

## 141. Linked List Cycle

### Two Pointers

time complexity: O(n)

space complexity: 1

```swift
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public var val: Int
 *     public var next: ListNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.next = nil
 *     }
 * }
 */

class Solution {
    func hasCycle(_ head: ListNode?) -> Bool {
        var slow = head, fast = head
        while fast != nil && fast?.next != nil {
            slow = slow?.next
            fast = fast?.next?.next
            if slow === fast {
                return true
            }
        }
        return false
    }
}
```

