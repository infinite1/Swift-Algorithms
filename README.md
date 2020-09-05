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

space complexity: O(1)

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

## 142. Linked List Cycle II

### Two Pointers

time complexity: O(n)

space complexity: O(1)

先找到交点，如果有环的话，在用两个指针指向head和交点，一次只移动一步，指针相遇的地方即为环的起始点

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
    func detectCycle(_ head: ListNode?) -> ListNode? {
        
        var head = head, intersect = getIntersect(head)
        if intersect == nil { return nil}
        
        while head !== intersect {
            head = head?.next
            intersect = intersect?.next
        }
        
        return head
        
    }
    
    private func getIntersect(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil {
            return nil
        }
        
        var slow = head, fast = head
        while fast != nil && fast?.next != nil {
            slow = slow?.next
            fast = fast?.next?.next
            if slow === fast {
                return slow
            }
        }
        
        return nil
    }
}
```

## 160. Intersection of Two Linked Lists

### Two Pointers

time complexity: O(m + n)

space complexity: O(1)

双指针便利两个链表A和B，如果指针P1先遍历完链表A，P1便指向链表B的head， 类似的，如果指针P2先遍历完，P2指向链表A的head。*如果两个链表无相交点，最终P1和P2都会为nil，因此跳出循环返回nil，说明没有相交点*

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
    func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
        if headA == nil || headB == nil {
            return nil
        }
        
        var pos1 = headA, pos2 = headB 
        while pos1 !== pos2 {
            if pos1 == nil {
                pos1 = headB
            } else {
                pos1 = pos1?.next
            }
            
            if pos2 == nil {
                pos2 = headA
            } else {
                pos2 = pos2?.next
            }
        }
        
        return pos1
    }
}
```

## 19. Remove Nth Node From End of List

### Two Pass Algorithm

先遍历得出链表的长度，再遍历删除（l - n + 1）个节点，*使用伪节点来避免讨论一些corner case，比如链表只有一个节点*。

time complexity: O(L)

space complexity: O(1)

```swift
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public var val: Int
 *     public var next: ListNode?
 *     public init() { self.val = 0; self.next = nil; }
 *     public init(_ val: Int) { self.val = val; self.next = nil; }
 *     public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
 * }
 */
class Solution {
    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        var pesudoNode = ListNode(0)
        pesudoNode.next = head
        
        var prev: ListNode? = pesudoNode
        let length = getLength(head)
        for _ in stride(from: 0, to: length-n, by: 1) {
            prev = prev?.next
        }
        prev?.next = prev?.next?.next
        return pesudoNode.next
    }
    
    private func getLength(_ head: ListNode?) -> Int {
        var cur = head
        var count = 0
        while cur != nil {
            cur = cur?.next
            count += 1
        }
        return count
    }
}
```

## 206. Reverse Linked List

### recursive approach

从head开始，将后面的链表看作已经排列好的链表，然后与head反转顺序。*每次递归调用后返回的都是链表的最后一个节点，或者说是反转链表的head*。

time complexity: O(n)

space complexity: O(n)

```swift
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public var val: Int
 *     public var next: ListNode?
 *     public init() { self.val = 0; self.next = nil; }
 *     public init(_ val: Int) { self.val = val; self.next = nil; }
 *     public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
 * }
 */
class Solution {
    func reverseList(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil { return head }
        var cur = head
        var node = reverseList(cur?.next)
        cur?.next?.next = cur
        cur?.next = nil
        return node
    }
}
```

## 203. Remove Linked List Elements

在head前添加一个伪head，这样的话可以避免讨论删除的数刚好在头部的corner case

time complexity: O(n)

space complexity: O(1)

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public var val: Int
 *     public var next: ListNode?
 *     public init() { self.val = 0; self.next = nil; }
 *     public init(_ val: Int) { self.val = val; self.next = nil; }
 *     public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
 * }
 */
class Solution {
    func removeElements(_ head: ListNode?, _ val: Int) -> ListNode? {
        var pesudoNode = ListNode(0)
        pesudoNode.next = head
        
        var prev: ListNode? = pesudoNode, cur = head
        while cur != nil {
            if cur?.val == val {
                prev?.next = cur?.next
                cur = cur?.next
            } else {
                cur = cur?.next
                prev = prev?.next
            } 
        }
        
        return pesudoNode.next
        
    }
}
```

