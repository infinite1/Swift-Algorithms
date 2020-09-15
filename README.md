# Swift-Algorithms
## 82. Remove Duplicates from Sorted List II
遍历重复元素时注意确保`cur?.next != nil`来应对如[1,1]这样的corner case
- time complexity: O(n)
- space complexity: O(1)
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
    func deleteDuplicates(_ head: ListNode?) -> ListNode? {
        var dummyHead: ListNode? = ListNode(0)
        dummyHead?.next = head
        var cur = head, prev = dummyHead
        while cur?.next != nil {
            if (prev?.next)!.val != (cur?.next)!.val {
                prev = cur
                cur = cur?.next
            } else {
                while cur?.next != nil && (prev?.next)!.val == (cur?.next)!.val {
                    cur = cur?.next
                }
                prev?.next = cur?.next
                cur = cur?.next
            }
        }
        return dummyHead?.next
    }
}
```
## 83. Remove Duplicates from Sorted List
- time complexity: O(N)
- space complexity: O(1)
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
    func deleteDuplicates(_ head: ListNode?) -> ListNode? {
        var cur = head
        while cur?.next != nil {
            if cur!.val == (cur?.next)!.val {
                cur?.next = cur?.next?.next
            } else {
                cur = cur?.next
            }
            
        }
        return head
    }
}
```
## 701. Insert into a Binary Search Tree
### Recursion
- time complexity: O(H), average O(logN), worst O(N)
- space complexity: O(H), average O(logN), worst O(N)
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    func insertIntoBST(_ root: TreeNode?, _ val: Int) -> TreeNode? {
        guard let root = root else { return TreeNode(val) }

        if val > root.val {
            root.right = insertIntoBST(root.right, val)
        } else {
            root.left = insertIntoBST(root.left, val)
        }

        return root
    }
}
```
### Iteration
- time complexity: O(H), average O(logN), worst O(N)
- space complexity: O(1)
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    func insertIntoBST(_ root: TreeNode?, _ val: Int) -> TreeNode? {
        var cur = root
        let toAdd = TreeNode(val)

        while cur != nil {
            if val > cur!.val {
                if let right = cur?.right {
                    cur = right
                } else {
                    cur?.right = toAdd
                    return root
                }
            } else {
                if let left = cur?.left {
                    cur = left
                } else {
                    cur?.left = toAdd
                    return root
                }
            }
        }

        return toAdd
    }
}
```
## 98. Validate Binary Search Tree
### Recursion
- time complexity: O(n)
- space complexity: O(n)
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func isValidBST(_ root: TreeNode?) -> Bool {
        return helper(root, Int.min, Int.max)
    }

    func helper(_ root: TreeNode?, _ low: Int, _ high: Int) -> Bool {
        guard let root = root else { return true }

        if root.val <= low || root.val >= high {
            return false
        }

        return helper(root.left, low, root.val) && helper(root.right, root.val, high)
    }
}
```
### Inorder iteration
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func isValidBST(_ root: TreeNode?) -> Bool {
        var root = root 

        var stack = [TreeNode?]()
        var inOrderPrev = Int.min

        while !stack.isEmpty || root != nil {
            while root != nil {
                stack.append(root)
                root = root?.left
            }

            let node = stack.removeLast()
            if node!.val <= inOrderPrev {
                return false
            }
            inOrderPrev = node!.val
            root = node?.right
        }

        return true
    }
}
```
- time complexity: O(n)
- space complexity: O(n)
## 103. Binary Tree Zigzag Level Order Traversal
### BFS
- time complexity: O(n)
- space complexity: O(n)
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func zigzagLevelOrder(_ root: TreeNode?) -> [[Int]] {
        var results = [[Int]]()
        guard let root = root else { return results }
        var queue = [TreeNode]()
        queue.append(root)
        var flag = true

        while !queue.isEmpty {
            var levelVal = [Int]()
            var size = queue.count
            while size > 0 {
                let node = queue.removeFirst()
                if flag {
                    levelVal.append(node.val)
                } else {
                    levelVal.insert(node.val, at: 0)
                }
                if let left = node.left {
                    queue.append(left)
                }
                if let right = node.right {
                    queue.append(right)
                }
                size -= 1
            }
            results.append(levelVal)
            flag = !flag
        }

        return results
    }
}

```
### DFS
- time complexity: O(n)
- space complexity: O(h)
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func zigzagLevelOrder(_ root: TreeNode?) -> [[Int]] {
        guard let root = root else { return [] }
        var results = [[Int]]()
        dfs(root, 0, &results)
        return results
    }

    func dfs(_ root: TreeNode, _ level: Int, _ results: inout [[Int]]) {
        if results.count == level {
            results.append([root.val])
        } else {
            if level % 2 != 0 {
                results[level].insert(root.val, at: 0)
            } else {
                results[level].append(root.val)
            }
        }

        if let left = root.left {
            dfs(left, level + 1, &results)
        }
        if let right = root.right {
            dfs(right, level + 1, &results)
        }
    }
}
```
## 107. Binary Tree Level Order Traversal II
### recursion
- time complexity: O(n)
- space complexity: O(h)
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    var results = [[Int]]()

    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        guard let root = root else { return results }
        traverseTree(root, 0)
        return results
    }

    func traverseTree(_ root: TreeNode, _ level: Int) {
        if results.count == level {
            results.append([Int]())
        }

        results[level].append(root.val)

        if let left = root.left {
            traverseTree(left, level + 1)
        }
        if let right = root.right {
            traverseTree(right, level + 1)
        }
    }
}
```
### Iteration
- time complexity: O(n)
- space complexity: O(n)
```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {

    func levelOrderBottom(_ root: TreeNode?) -> [[Int]] {
        var results = [[Int]]()
        guard let root = root else { return results }
        var queue = [TreeNode]()
        queue.append(root)

        while !queue.isEmpty {
            var size = queue.count
            var levelVal = [Int]()
            while size > 0 {
                let node = queue.removeFirst()
                levelVal.append(node.val)
                if let left = node.left {
                    queue.append(left)
                }
                if let right = node.right {
                    queue.append(right)
                }
                size -= 1
            }
            results.append(levelVal)
        }

        return results.reversed()
    }
}
```
## 124. Binary Tree Maximum Path Sum

time complexity: O(n)

space complexity: O(h)

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    var minSum = Int.min

    func maxPathSum(_ root: TreeNode?) -> Int {
        maxGain(root)
        return minSum
    }

    func maxGain(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }

        let left = max(maxGain(root.left), 0)
        let right = max(maxGain(root.right), 0)

        let sum = left + right + root.val
        minSum = max(minSum, sum)

        return root.val + max(left, right)
    }
}
```



## 110. Balanced Binary Tree

### top-down recursion

time complexity: O(n^2), height会被重复调用因此耗时较高
space complexity: O(h) 

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func isBalanced(_ root: TreeNode?) -> Bool {
        guard let root = root else { return true }
        return (abs(getHeight(root.left) - getHeight(root.right)) <= 1) && isBalanced(root.left) && isBalanced(root.right)
    }

    func getHeight(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        return 1 + max(getHeight(root.left), getHeight(root.right))
    }
}
```

### bottom-up recursion

time complexity: O(n)

space complexity: O(h) 

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func isBalanced(_ root: TreeNode?) -> Bool {
        return height(root) >= 0
    }

    func height(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        let left = height(root.left), right = height(root.right)
        if left == -1 || right == -1 || abs(left - right) > 1 {
            return -1
        }
        return 1 + max(left, right)
    }
}
```



## 237. Delete Node in a Linked List

time complexity: O(1)

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
    func deleteNode(_ node: ListNode?) {
        node!.val = (node?.next)!.val
        node?.next = node?.next?.next
    }
}
```



## 231. Power of Two

### dividing 2 approach

time complexity: O(log(n))

space complexity: O(1) 

```swift
class Solution {
    func isPowerOfTwo(_ n: Int) -> Bool {
        var n = n
        if n == 0 { return false }
        while n % 2 == 0 {
            n /= 2
        }
        return n == 1
    }
}
```



## 169. Majority Element

### hash table approach

time complexity: O(n)

space complexity: O(n) 

```swift
class Solution {
    func majorityElement(_ nums: [Int]) -> Int {
        var table = [Int: Int]()
        for i in nums {
            if table[i] == nil {
                table[i] = 0
            }
            table[i]! += 1
        }

        for (key, value) in table {
            if value > nums.count / 2 {
                return key
            }
        }

        return -1

    }
}
```



## 136. Single Number

### XOR approach

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        var res = 0
        for i in nums {
            res ^= i
        }
        return res
    }
}
```



## 9. Palindrome Number

time complexity: O(log(x))

space complexity: O(n) 

```swift
class Solution {
    func isPalindrome(_ x: Int) -> Bool {
        if x < 0 { return false }
        var rev = 0, X = x 
        while X != 0 {
            let pop = X % 10
            X /= 10
            rev = rev * 10 + pop
        }

        return rev == x
    }
}
```



## 7. Reverse Integer

check overflow beforehand because `res = res * 10 + pop` could cause overflow

time complexity: O(log(x))

space complexity: O(n) 

```swift
class Solution {
    func reverse(_ x: Int) -> Int {
        var res = 0, X = x
        while X != 0 {
            let pop = X % 10
            X /= 10

            if res > Int32.max / 10 || (res == Int32.max / 10 && pop > Int32.max % 10) {
                return 0
            }

            if res < Int32.min / 10 || (res == Int32.min / 10 && pop < Int32.min % 10) {
                return 0
            } 
            res = res * 10 + pop
        }
        return res
    }
}
```



## 448. Find All Numbers Disappeared in an Array

### hash map approach

time complexity: O(n)

space complexity: O(n) 

```swift
class Solution {
    func findDisappearedNumbers(_ nums: [Int]) -> [Int] {
        if nums.isEmpty { return [] }
        
        var table = [Int: Int]()
        var results = [Int]()
        
        for i in nums {
            table[i] = 1
        }
        for i in 0 ..< nums.count {
            let n = i + 1
            if table[n] != 1 {
                results.append(n)
            }
        }
        return results
    }
}
```



## 414. Third Maximum Number

### use set and delete max number

time complexity: O(n)

space complexity: O(n) 

```swift
class Solution {
    func thirdMax(_ nums: [Int]) -> Int {
        var uniqueNums = Set(nums)
        
        if uniqueNums.count < 3 {
            return uniqueNums.max()!
        }
        
        var maxNum = uniqueNums.max()!
        uniqueNums.remove(maxNum)
        maxNum = uniqueNums.max()!
        uniqueNums.remove(maxNum)
        
        return uniqueNums.max()!
    }
}


```



## 487. Max Consecutive Ones II

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func findMaxConsecutiveOnes(_ nums: [Int]) -> Int {
        var start = 0, maxLength = 0, maxOnes = 0
        for end in 0 ..< nums.count {
            if nums[end] == 1 {
                maxOnes += 1
            }
            
            if end - start + 1 - maxOnes > 1 {
                if nums[start] == 1 {
                   maxOnes -= 1 
                }
                start += 1
            }
            
            maxLength = max(maxLength, end - start + 1)
        }
        return maxLength
    }
}
```



## 1051. Height Checker

time complexity: O(nlogn)

space complexity: O(n) 

```swift
class Solution {
    func heightChecker(_ heights: [Int]) -> Int {
        var sortedArr = heights.sorted()
        var p1 = 0, p2 = 0, num = 0
        while p1 < heights.count && p2 < heights.count {
            if sortedArr[p1] != heights[p2] {
                num += 1
            }
            p1 += 1
            p2 += 1
        }
        return num
    }
}
```



## 905. Sort Array By Parity

### two iteration

time complexity: O(n)

space complexity: O(n) 

```swift
class Solution {
    func sortArrayByParity(_ A: [Int]) -> [Int] {
        var arr = [Int]()
        for i in A {
            if i % 2 == 0 {
                arr.append(i)
            }
        }
        for i in A {
            if i % 2 != 0 {
                arr.append(i)
            }
        }
        return arr
    }
}
```



## 283. Move Zeroes

### two iteration

第一次遍历把非0元素全部往左挪，第二次遍历把数组剩下的位置用0填充

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func moveZeroes(_ nums: inout [Int]) {
        var j = 0
        for i in 0 ..< nums.count {
            if nums[i] != 0 {
                nums[j] = nums[i]
                j += 1
            }
        }
        for i in j ..< nums.count {
            nums[i] = 0
        }
    }
}
```



## 1299. Replace Elements with Greatest Element on Right Side

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func replaceElements(_ arr: [Int]) -> [Int] {
        var arr = arr
        var p = arr.count - 1, maxVal = -1
        while p >= 0 {
            let prev = arr[p]
            arr[p] = maxVal
            maxVal = max(maxVal, prev)
            p -= 1
        }
        return arr
    }
}
```



## 941. Valid Mountain Array

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func validMountainArray(_ A: [Int]) -> Bool {
        let l = A.count
        var i = 0
        
        while i + 1 < l && A[i] < A[i + 1] {
            i += 1
        }
        
        // check if array is in descending or in ascending order
        if i == 0 || i == l - 1 {
            return false
        }
        
        // if two element are the same, i won't arrive at l - 1
        while i + 1 < l && A[i] > A[i + 1] {
            i += 1
        }
        
        return i == l - 1
        
    }
}
```



## 1346. Check If N and Its Double Exist

### hashtable

 time complexity: O(n)

space complexity: O(n) 

```swift
class Solution {
    func checkIfExist(_ arr: [Int]) -> Bool {
        var table = [Int: Int]()
        
        for i in arr {
            if table[i * 2] != nil || (i % 2 == 0 && table[i / 2] != nil) {
                return true
            } else {
                table[i] = 1
            }
        }
        
        return false
    }
}
```



## 26. Remove Duplicates from Sorted Array

 time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func removeDuplicates(_ nums: inout [Int]) -> Int {
        if nums.count == 0 { return 0 }
        var i = 0
        for j in 1 ..< nums.count {
            if nums[j] != nums[i] {
                i += 1
                nums[i] = nums[j]
            }
        }
        return i + 1
    }
}
```



## 27. Remove Element

### two pointers

需要移除的元素很多时

 time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func removeElement(_ nums: inout [Int], _ val: Int) -> Int {
        var i = 0, j = 0
        for j in 0 ..< nums.count {
            if nums[j] != val {
                nums[i] = nums[j]
                i += 1
            }
        }
        return i
    }
}
```



## 88. Merge Sorted Array

### two pointer from beginning

time complexity: O(m + n)

space complexity: O(m) 

```swift
class Solution {
    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        let nums1Copy = nums1[0..<m]
        nums1 = []
        var p1 = 0
        var p2 = 0
        while p1 < m && p2 < n {
            if nums1Copy[p1] < nums2[p2] {
                nums1.append(nums1Copy[p1])
                p1 += 1
            } else {
                nums1.append(nums2[p2])
                p2 += 1
            }
        }
        
        if p1 < m {
            nums1 += nums1Copy[p1...]
        }
        if p2 < n {
            nums1 += nums2[p2...]
        }
    }
}


```

### two pointer from end

time complexity: O(n + m)

space complexity: O(1) 

```swift
class Solution {
    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        var p1 = m - 1, p2 = n - 1
        var p = nums1.count - 1
        while p1 >= 0 && p2 >= 0 {
            if nums1[p1] > nums2[p2] {
                nums1[p] = nums1[p1]
                p1 -= 1
            } else {
                nums1[p] = nums2[p2]
                p2 -= 1
            }
            p -= 1
        }
        nums1[...p2] = nums2[...p2]
    }
}


```



## 1089. Duplicate Zeros

两次遍历，第一次遍历找到多少个元素要被丢弃，第二次遍历从后往前把元素向后移

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func duplicateZeros(_ arr: inout [Int]) {
        var possibleDuplicate = 0
        var length = arr.count - 1
        for left in 0 ..< arr.count {
            if left > length - possibleDuplicate {
                break
            }
            
            if arr[left] == 0 {
                if left == length - possibleDuplicate {
                    arr[length] = 0
                    length -= 1
                    break
                }
                possibleDuplicate += 1
            }
        }
        
        var last = length - possibleDuplicate
        
        for i in stride(from: last, to: -1, by: -1) {
            if arr[i] == 0 {
                arr[i + possibleDuplicate] = 0
                possibleDuplicate -= 1
                arr[i + possibleDuplicate] = 0
            } else {
                arr[i + possibleDuplicate] = arr[i]
            }
        }
    }
}
```



## 977. Squares of a Sorted Array

### sorting

use built-in sort function

time complexity: O(nlogn)

space complexity: O(n) 

### two pointer

time complexity: O(n)

space complexity: O(n) 

```swift
class Solution {
    func sortedSquares(_ A: [Int]) -> [Int] {
        var arr = [Int]()
        
        var j = 0
        while j < A.count && A[j] < 0 {
            j += 1
        }
        var i = j - 1
        
        while i >= 0 && j < A.count {
            let val1 = A[i] * A[i]
            let val2 = A[j] * A[j]
            if val1 < val2 {
                arr.append(val1)
                i -= 1
            } else {
                arr.append(val2)
                j += 1
            }
        }
        
        while i >= 0 {
            arr.append(A[i] * A[i])
            i -= 1
        }
        
        while j < A.count {
            arr.append(A[j] * A[j])
            j += 1
        }
        
        return arr
    }
}
```



## 1295. Find Numbers with Even Number of Digits

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func findNumbers(_ nums: [Int]) -> Int {
        var digitsCount = nums.map { String($0).count }
        var count = 0
        for i in digitsCount {
            if i % 2 == 0 {
                count += 1
            }
        }
        return count
    }
}
```



## 485. Max Consecutive Ones

在for loop中无法更新最后一次的maxCount，所以在循环结束后需要手动更新一次maxCount

time complexity: O(n)

space complexity: O(1) 

```swift
class Solution {
    func findMaxConsecutiveOnes(_ nums: [Int]) -> Int {
        var start = 0
        var maxCount = 0
        var count = 0
        for end in 0 ..< nums.count {
            if nums[end] == 1 {
                count += 1
            } else {
                start = end + 1
                maxCount = max(maxCount, count)
                count = 0
            }
        }
        
        return max(maxCount, count)
    }
}
```



## 297. Serialize and Deserialize Binary Tree

DFS

for both serialisation and deserialisation function

time complexity: O(n)

space complexity: O(n) 

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */

class Codec {
    func serialize(_ root: TreeNode?) -> String {
        var ans = ""
        return rserialize(root, &ans)
    }
    
    private func rserialize(_ root: TreeNode?, _ str: inout String) -> String {
        if let root = root {
            str += String(root.val) + ","
            str = rserialize(root.left, &str)
            str = rserialize(root.right, &str)
        } else { 
            str += "nil," 
        }
        
        return str
    }
    
    func deserialize(_ data: String) -> TreeNode? {
        var dataList = data.split(separator: ",").map { String($0) }
        return rdeserialize(&dataList)
    }
    
    func rdeserialize(_ data: inout [String]) -> TreeNode? {
        if data[0] == "nil" {
            data.removeFirst()
            return nil
        }
        
        let nodeVal = Int(data[0])
        let root = TreeNode(nodeVal!)
        data.removeFirst()
        
        root.left = rdeserialize(&data)
        root.right = rdeserialize(&data)
        return root
    }
}

// Your Codec object will be instantiated and called as such:
// var codec = Codec()
// codec.deserialize(codec.serialize(root))
```



## 236. Lowest Common Ancestor of a Binary Tree

### recursion

time complexity: O(n)

space complexity: O(h), h is the height of tree, worst case O(n)

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */

class Solution {
    var ans: TreeNode?
    
    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        recursTree(root, p, q)
        return ans
    }
    
    func recursTree(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> Bool {
        guard let root = root else { return false }
        
        let left = recursTree(root.left, p, q)
        let right = recursTree(root.right, p, q)
        
        if (left && right) || ((left || right) && (root.val == p!.val || root.val == q!.val)) {
            ans = root
        }
        
        return left || right || (root.val == p!.val || root.val == q!.val)
    }
}
```

### use hashtable to store root node

time complexity: O(n)

space complexity: O(n)

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */

class Solution {
    var parents = [Int: TreeNode?]()
    var visited = [Int]()

    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        dfs(root)
        var p = p, q = q
        while p != nil {
            visited.append(p!.val)
            p = parents[p!.val] ?? nil
        }

        while q != nil {
            if visited.contains(q!.val) {
                return q
            }
            q = parents[q!.val] ?? nil
        }
        return nil
    }

    private func dfs(_ root: TreeNode?) {
        if let left = root?.left {
            parents[left.val] = root
            dfs(left)
        }
        if let right = root?.right {
            parents[right.val] = root
            dfs(right)
        }
    }
}
```



## 117. Populating Next Right Pointers in Each Node II

### Level Order Traversal

time complexity: O(n)

space complexity: O(n), depending on the level that have max number of nodes

```swift
/**
 * Definition for a Node.
 * public class Node {
 *     public var val: Int
 *     public var left: Node?
 *     public var right: Node?
 *	   public var next: Node?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *         self.next = nil
 *     }
 * }
 */

class Solution {
    func connect(_ root: Node?) -> Node? {
        guard let root = root else { return nil }
        
        var queue = [Node?]()
        queue.append(root)
        
        while !queue.isEmpty {
            let size = queue.count
            
            for i in 0 ..< size {
                let node = queue.removeFirst()
                if i < size - 1 {
                    node?.next = queue[0]
                }
                
                if let left = node?.left {
                    queue.append(left)
                }
                
                if let right = node?.right {
                    queue.append(right)
                }
            }
            
        }
        
        return root
    }
}
```



## 116. Populating Next Right Pointers in Each Node

time complexity: O(n)

space complexity: O(n), depending on the level that have max number of nodes

```swift
/**
 * Definition for a Node.
 * public class Node {
 *     public var val: Int
 *     public var left: Node?
 *     public var right: Node?
 *	   public var next: Node?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *         self.next = nil
 *     }
 * }
 */

class Solution {
    func connect(_ root: Node?) -> Node? {
        guard let root = root else { return nil }
        
        var queue = [Node?]()
        queue.append(root)
        
        while !queue.isEmpty {
            let size = queue.count 
            for i in 0 ..< size {
                let node = queue.removeFirst()
                if i < size - 1 {
                    node?.next = queue[0]
                }
                
                if let left = node?.left {
                    queue.append(left)
                }
                if let right = node?.right {
                    queue.append(right)
                }
            }
        }
        
        return root
    }
}
```



## 105. Construct Binary Tree from Preorder and Inorder Traversal

similar to 106, the only difference is we call `node.left` first because of preorder.

time complexity: O(n), master theorem

space complexity: O(n) 

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    var preOrderIndex = 0
    var preOrder = [Int]()
    var inOrder = [Int]()
    var indexTable = [Int: Int]()
    
    func buildTree(_ preorder: [Int], _ inorder: [Int]) -> TreeNode? {
        preOrder = preorder
        inOrder = inorder
        for i in 0 ..< inorder.count {
            indexTable[inorder[i]] = i
        }
        return helper(0, inorder.count - 1)
    }
    
    func helper(_ left: Int, _ right: Int) -> TreeNode? {
        if left > right {
            return nil
        }
        
        let nodeVal = preOrder[preOrderIndex]
        let node = TreeNode(nodeVal)
        let index = indexTable[nodeVal]!
        
        preOrderIndex += 1
        node.left = helper(left, index - 1)
        node.right = helper(index + 1, right)
        
        return node
    }
}
```



## 106. Construct Binary Tree from Inorder and Postorder Traversal

We call `root.right = helper(index + 1, right)` because postorder is left-right-node, so each time the  postIndex posts to a node which is right of the previous position.

time complexity: O(n), master theorem

space complexity: O(n) 

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    var indexTable = [Int: Int]()
    var postOrder = [Int]()
    var postIndex = 0
    var inOrder = [Int]()

    func buildTree(_ inorder: [Int], _ postorder: [Int]) -> TreeNode? {
        postOrder = postorder
        inOrder = inorder
        postIndex = postorder.count - 1
        for i in 0 ..< inorder.count {
            indexTable[inorder[i]] = i
        }
        return helper(0, inorder.count - 1)
    }
    
    func helper(_ left: Int, _ right: Int) -> TreeNode? {
        if left > right {
            return nil
        }
        
        let rootVal = postOrder[postIndex]
        let root = TreeNode(rootVal)
        
        let index = indexTable[rootVal]!
        
        postIndex -= 1
        root.right = helper(index + 1, right)
        root.left = helper(left, index - 1)
        return root
    }
}
```



## 250. Count Univalue Subtrees

### DFS

time complexity: O(n)

space complexity: O(n) worst case when the tree is completely unbalanced, average O(H) or O(logn) if the tree is completely balanced, where H is tree height

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    var count = 0
    
    func countUnivalSubtrees(_ root: TreeNode?) -> Int {
        guard let root = root else { return count }
        checkUnival(root)
        return count
    }
    
    private func checkUnival(_ root: TreeNode?) -> Bool {
        if root?.left == nil && root?.right == nil {
            count += 1
            return true
        }
        
        var isUnival = true
        
        // check if two children subtrees are unival
        if root?.left != nil {
            isUnival = checkUnival(root?.left) && isUnival && (root?.left)!.val == (root!.val)
        }
        if root?.right != nil {
            isUnival = checkUnival(root?.right) && isUnival && (root?.right)!.val == (root!.val)
        }
        
        if !isUnival { return false }
        count += 1
        return true
    }
}
```



## 112. Path Sum

### recursion

*the problem asks for root-to-leaf path sum*

time complexity: O(n)

space complexity: O(n) worst case when the tree is completely unbalanced, average O(H) or O(logn) if the tree is completely balanced, where H is tree height

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    func hasPathSum(_ root: TreeNode?, _ sum: Int) -> Bool {
        guard let root = root else { return false }
        if root.left == nil && root.right == nil { return root.val == sum }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val)
    }
}
```



## 101. Symmetric Tree

### recursion

time complexity: O(n)

space complexity: O(n) worset case, average O(H) or O(logn) if the tree is completely balanced, where H is tree height

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    func isSymmetric(_ root: TreeNode?) -> Bool {
        return isMirror(root, root)
    }
    
    func isMirror(_ node1: TreeNode?, _ node2: TreeNode?) -> Bool {
        if node1 == nil && node2 == nil {
            return true
        }
        if node1 == nil || node2 == nil {
            return false
        }
        return node1!.val == node2!.val && isMirror(node1?.left, node2?.right) && isMirror(node1?.right, node2?.left)
    }
    
}
```



## 104. Maximum Depth of Binary Tree

### Recursion

time complexity: O(n)

space complexity: O(n) worset case, average O(H) or O(logn) if the tree is completely balanced, where H is tree height

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    func maxDepth(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        return 1 + max(maxDepth(root.left), maxDepth(root.right))
    }
}
```

### BFS

time complexity: O(n)

space complexity: O(s). where s is the maximum elements in a level

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func maxDepth(_ root: TreeNode?) -> Int {
        guard let root = root else { return 0 }
        var queue = [TreeNode?]()
        queue.append(root)
        var depth = 0
        while !queue.isEmpty {
            var i = queue.count
            while i > 0 {
                let node = queue.removeFirst()
                if let left = node?.left {
                    queue.append(left)
                }
                if let right = node?.right {
                    queue.append(right)
                }
                i -= 1
            }
            depth += 1
        }
        return depth
    }
}
```



## 102. Binary Tree Level Order Traversal

### recursion

time complexity: O(n)

space complexity: O(n)

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    var results = [[Int]]()

    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        guard let root = root else { return results }
        traverseTree(root, 0)
        return results
    }

    func traverseTree(_ root: TreeNode, _ level: Int) {
        if results.count == level {
            results.append([Int]())
        }

        results[level].append(root.val)

        if let left = root.left {
            traverseTree(left, level + 1)
        }
        if let right = root.right {
            traverseTree(right, level + 1)
        }
    }
}
```

### iteration

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.left = nil
 *         self.right = nil
 *     }
 * }
 */
class Solution {
    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        guard let root = root else { return [] }
        var queue = [TreeNode?]()
        var results = [[Int]]()
        queue.append(root)
        while !queue.isEmpty {
            var size = queue.count
            var levelVal = [Int]()
            while size > 0 {
                let node = queue.removeFirst()
                levelVal.append(node!.val)
                if let left = node?.left {
                    queue.append(left)
                }
                if let right = node?.right {
                    queue.append(right)
                }
                size -= 1
            }
            results.append(levelVal)
        }
        return results
    }
}
```

time complexity: O(n)
space complexity: O(n)

## 145. Binary Tree Postorder Traversal

time complexity: O(n)

space complexity: O(n) worset case, average O(H), where H is tree height

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    func postorderTraversal(_ root: TreeNode?) -> [Int] {
        var results = [Int]()
        
        guard let root = root else {
            return []
        }
        
        results += postorderTraversal(root.left)
        results += postorderTraversal(root.right)
        results.append(root.val)
        
        return results
    }
}
```



## 94. Binary Tree Inorder Traversal

time complexity: O(n)

space complexity: O(n) worset case, average O(H), where H is tree height

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    func inorderTraversal(_ root: TreeNode?) -> [Int] {
        var results = [Int]()
        
        guard let root = root else {
            return []
        }
        
        results += inorderTraversal(root.left)
        results.append(root.val)
        results += inorderTraversal(root.right)
        
        return results
    }
}
```



## 144. Binary Tree Preorder Traversal

### recursive approach

time complexity: O(n)

space complexity: O(n) worset case, average O(H), where H is tree height

```swift
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public var val: Int
 *     public var left: TreeNode?
 *     public var right: TreeNode?
 *     public init() { self.val = 0; self.left = nil; self.right = nil; }
 *     public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
 *     public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
 *         self.val = val
 *         self.left = left
 *         self.right = right
 *     }
 * }
 */
class Solution {
    
    
    func preorderTraversal(_ root: TreeNode?) -> [Int] {
        var results = [Int]()
        
        guard let root = root else {
            return []
        }
        
        results.append(root.val)
        results += preorderTraversal(root.left)
        results += preorderTraversal(root.right)
        
        return results
    }
}
```



## 61. Rotate List

time complexity: O(n)

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
    func rotateRight(_ head: ListNode?, _ k: Int) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }
        
        // connect tail into head
        var oldTail = head
        var n = 1
        while oldTail?.next != nil {
            oldTail = oldTail?.next
            n += 1
        }
        oldTail?.next = head
        
        // find new tail and new head
        var newTail = head
        for _ in 0 ..< n - k % n - 1 {
            newTail = newTail?.next
        }
        let newHead = newTail?.next
        
        newTail?.next = nil
        return newHead
    }
}
```



## 138. Copy List with Random Pointer

### recursive approach

*the head can be any node in the list*

time complexity: O(n)

space complexity: O(n), 字典和递归的stack都会占用n的复杂度

```swift
/**
 * Definition for a Node.
 * public class Node {
 *     public var val: Int
 *     public var next: Node?
 *     public var random: Node?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.next = nil
 *    	   self.random = nil
 *     }
 * }
 */

class Solution {
    var visited = [Node?: Node?]()
    
    func copyRandomList(_ head: Node?) -> Node? {
        if head == nil {
            return nil
        }
        
        if visited[head] != nil {
            return visited[head]!
        }
        
        var copyNode = Node(head!.val)
        
        visited[head] = copyNode
        
        copyNode.next = copyRandomList(head?.next)
        copyNode.random = copyRandomList(head?.random)
        
        return copyNode
    
    }
}
```



## 708. Insert into a Sorted Circular Linked List

双指针遍历

1. insert between small and large value
2. insert between tail and head if the value is larger than tail or smaller than head
3. insert after head if all elements are the same

time complexity: O(n)

space complexity: O(1）

```swift
/**
 * Definition for a Node.
 * public class Node {
 *     public var val: Int
 *     public var next: Node?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.next = nil
 *     }
 * }
 */

class Solution {
    func insert(_ head: Node?, _ insertVal: Int) -> Node? {
        if head == nil {
            let node = Node(insertVal)
            node.next = node
            return node
        }
        
        var prev = head, cur = head?.next
        var toInsert = false
        
        repeat {
            if prev!.val <= insertVal && insertVal <= cur!.val {
                toInsert = true
            } else if prev!.val > cur!.val  {
                // if insertVal is larger than tail or smaller than head
                // insert between tail and head
                if insertVal < cur!.val || insertVal > prev!.val {
                    toInsert = true
                }
                
            }
            
            if toInsert {
                let node = Node(insertVal)
                prev?.next = node
                node.next = cur
                return head
            }
            
            prev = cur
            cur = cur?.next
        } while prev !== head
        
        // if all elements are the same, insert after head
        let node = Node(insertVal)
        prev?.next = node
        node.next = cur
        
        return head
    }
}
```



## 430. Flatten a Multilevel Doubly Linked List

### DFS by recursion

可以将这个问题看成是求DFS的preorder遍历

time complexity: O(n), 每个节点都会被访问一遍

space complexity: O(n), 如果是不平衡二叉树，所有的节点都有child，那么递归空间就会占用n的复杂度，如果所有节点都没有child，相当于tail recursion，只占用1的复杂度

```swift
/**
 * Definition for a Node.
 * public class Node {
 *     public var val: Int
 *     public var prev: Node?
 *     public var next: Node?
 *     public var child: Node?
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.prev = nil
 *         self.next = nil
 *         self.child  = nil
 *     }
 * }
 */

class Solution {
    func flatten(_ head: Node?) -> Node? {
        guard let head = head else { return nil}
        
        let pesudoHead = Node(0)
        flattenDFS(pesudoHead, head)
        
        pesudoHead.next?.prev = nil
        
        return pesudoHead.next
        
    }
    
    func flattenDFS(_ prev: Node?, _ curr: Node?) -> Node? {
        guard let curr = curr else { return prev }
        
        curr.prev = prev
        prev?.next = curr
        
        let tempNext = curr.next
        let tail = flattenDFS(curr, curr.child)
        curr.child = nil
        
        return flattenDFS(tail, tempNext)
        
    }
}
```



## 2. Add Two Numbers

使用伪head，双指针和基本数学加法(val1+val2+carry)来构建链表

time complexity: O(max(m, n))

space complexity: O(max(m, n)), 新链表长度为max(m, n)

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
    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        var pesudoHead = ListNode(0)
        var p1 = l1, p2 = l2, prev = pesudoHead
        var carry = 0
        while p1 != nil || p2 != nil {
            let val1 = p1 != nil ? p1!.val : 0
            let val2 = p2 != nil ? p2!.val : 0
            let sum = val1 + val2 + carry
            carry = sum / 10
            let toAdd = ListNode(sum % 10)
            prev.next = toAdd
            prev = toAdd
            p1 = p1?.next
            p2 = p2?.next
        }
        
        if carry == 1 {
            prev.next = ListNode(carry)
        }
        
        return pesudoHead.next
    }
}
```



## 707. Design Linked List

### Use Single Linked List

使用哨兵节点当作伪head，这样的话就不需要讨论list为空的情况，而且可以把addAtHead和addAtTail都简化为addAtIndex的特殊形式。*链表的head是哨兵节点，真正的head可以用head.next访问*

time complexity: O(k) of addAtIndex, get and deleteIndex, O(1) for addAtHead, O(n) for addAtTail

space complexity: O(1)

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

### Double Linked List

和单链表类似，在链表末尾添加伪尾节点，每次插入或删除节点时，可以从head或者tail遍历，提升访问速度。*在addAtIndex里，通过找到当前节点的predecessor节点和successor节点，再进行节点的插入，不过需要仔细考虑边界问题*

time complexity: O(min(k, n - k)) of addAtIndex, get and deleteIndex, O(1) for getHead and getTail

space complexity: O(1)

```swift
class Node {
    let val: Int
    var next: Node?
    var prev: Node?
    init(_ val: Int) {
        self.val = val
    } 
}

class MyLinkedList {
    
    
    private var head: Node?
    private var tail: Node?
    private var size: Int

    /** Initialize your data structure here. */
    init() {
        head = Node(0)
        tail = Node(0)
        head?.next = tail
        tail?.prev = head
        size = 0
    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    func get(_ index: Int) -> Int {
        if index < 0 || index >= self.size {
            return -1
        }
        
        var curr: Node?
        if index < self.size - index - 1 {
            // start from head
            curr = self.head
            for _ in stride(from: 0, to: index + 1, by: 1) {
                curr = curr?.next
            }
            
        } else {
            // start from tail
            curr = self.tail
            for _ in stride(from: 0, to: self.size - index, by: 1) {
                curr = curr?.prev
            }
        }
        
        return (curr?.val)!
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    func addAtHead(_ val: Int) {
        self.addAtIndex(0, val)
    }
    
    /** Append a node of value val to the last element of the linked list. */
    func addAtTail(_ val: Int) {
        self.addAtIndex(self.size, val)
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    func addAtIndex(_ index: Int, _ val: Int) {
        // if index is greater than linkedlist size
        // or index < 0, the node will not be inserted
        if index > self.size || index < 0 { return }
        
        self.size += 1
        var pred: Node?, succ: Node?
        if index < self.size - index - 1 {
            // start from head
            pred = self.head
            for _ in stride(from: 0, to: index, by: 1) {
                pred = pred?.next
            }
            succ = pred?.next
            
        } else {
            // start from tail
            succ = self.tail
            for _ in stride(from: 0, to: self.size - index - 1, by: 1) {
                succ = succ?.prev
            }
            pred = succ?.prev
        }
        
        // create new node with val
        let toAdd = Node(val)

        // connect with pred and succ
        toAdd.prev = pred
        toAdd.next = succ
        pred?.next = toAdd
        succ?.prev = toAdd
        
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    func deleteAtIndex(_ index: Int) {
        if index < 0 || index >= self.size { return }
        
        var pred: Node?, succ: Node?
        if index < self.size - index - 1 {
            // start from head
            pred = self.head
            for _ in stride(from: 0, to: index, by: 1) {
                pred = pred?.next
            }
            succ = pred?.next?.next
            
        } else {
            // start from tail
            succ = self.tail
            for _ in stride(from: 0, to: self.size - index - 1, by: 1) {
                succ = succ?.prev
            }
            pred = succ?.prev?.prev
        }
        
        self.size -= 1
        pred?.next = succ
        succ?.prev = pred
        
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
### Recursion
从head开始，将后面的链表看作已经排列好的链表，然后与head反转顺序。*每次递归调用后返回的都是链表的最后一个节点，或者说是反转链表的head*。
- time complexity: O(N)
- space complexity: O(N)
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
    func reverseList(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }

        var newHead = reverseList(head?.next)
        head?.next?.next = head
        head?.next = nil

        return newHead
    }
}
```
### Iteration
- time complexity: O(N)
- space complexity: O(1) 
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
    func reverseList(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }

        var prev: ListNode? = nil, cur = head
        while cur != nil {
            let temp = cur?.next
            cur?.next = prev
            prev = cur
            cur = temp
        }

        return prev
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

## 328. Odd Even Linked List

even?.next != nil确保odd?.next不会指向nil，如果odd的下一个节点是nil，那么由odd = odd?.next也会指向nil，连接odd链表和even链表时就会变成nil.next = evenHead，这当然不是我们想要的

time complexity: O(n)

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
    func oddEvenList(_ head: ListNode?) -> ListNode? {
        if head == nil { return nil }
        var odd = head, even = head?.next, evenHead = even
        while even != nil && even?.next != nil {
            odd?.next = even?.next
            odd = odd?.next
            even?.next = odd?.next
            even = even?.next
        }
        odd?.next = evenHead
        return head
    }
}
```

## 234. Palindrome Linked List

### Copy Linked List into array and use two pointers approach

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
    func isPalindrome(_ head: ListNode?) -> Bool {
        if head == nil || head?.next == nil {
            return true
        }
        var arr = toArray(head)
        print(arr)
        var left = 0, right = arr.count - 1
        while left < right {
            if arr[left] != arr[right] {
                return false
            } else {
                left += 1
                right -= 1
            }
        }
        return true
    
    }
    
    private func toArray(_ head: ListNode?) -> [Int] {
        var cur = head
        var arr = [Int]()
        while cur != nil {
            arr.append(cur!.val)
            cur = cur?.next
        }
        return arr
    }
}
```

## 21. Merge Two Sorted Lists

### Iteration Approach

使用伪head节点来方便返回head

time complexity: O(m+n)

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
    func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? { 
        var p1 = l1, p2 = l2
        var pesudoHead = ListNode(0)
        var prev = pesudoHead
        while let cur1 = p1, let cur2 = p2 {
            let val1 = cur1.val, val2 = cur2.val
            if val1 <= val2 {
                prev.next = cur1
                p1 = cur1.next
            } else {
                prev.next = p2
                p2 = cur2.next
            }
            prev = prev.next!
        }
        
        prev.next = p1 ?? p2
        
        return pesudoHead.next
    }
}
```

