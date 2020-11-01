# LeetCode-Swift-Algorithms
## 516. Longest Palindromic Subsequence
### DP
- time complexity: O(n^2)
- space complexity: O(n^2)
```swift
class Solution {
    func longestPalindromeSubseq(_ s: String) -> Int {
        var sArr = [Character](s)
        let n = sArr.count
        var dp = Array(repeating: Array(repeating: 0, count: n), count: n)
        for i in 0..<n {
            dp[i][i] = 1
        }
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in i + 1..<n {
                if sArr[i] == sArr[j] {
                    dp[i][j] = dp[i + 1][j - 1] + 2
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
                }
            }
        }
        return dp[0][n - 1]
    }
}
```
### DP + Compression
- time complexity: O(n^2)
- space complexity: O(n)
```swift
class Solution {
    func longestPalindromeSubseq(_ s: String) -> Int {
        var sArr = [Character](s)
        let n = sArr.count
        var dp = Array(repeating: 1, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var pre = 0
            for j in i + 1..<n {
                let tmp = dp[j]
                if sArr[i] == sArr[j] {
                    dp[j] = pre + 2
                } else {
                    dp[j] = max(dp[j], dp[j - 1])
                }
                pre = tmp
            }
        }
        return dp[n - 1]
    }
}
```
## 354. Russian Doll Envelopes
### DP
- time complexity: O(n^2)
- space complexity: O(n)
```swift
class Solution {
    func maxEnvelopes(_ envelopes: [[Int]]) -> Int {
        let n = envelopes.count
        let sortedEnvelopes = envelopes.sorted {
            return $0[0] == $1[0] ? $0[1] > $1[1] : $0[0] < $1[0]
        }
        var heights = [Int]()
        for envelope in sortedEnvelopes {
            heights.append(envelope[1]);
        }

        var dp = Array(repeating: 1, count: n);
        for i in 0..<n {
            for j in 0..<i {
                if heights[i] > heights[j] {
                    dp[i] = max(dp[i], dp[j] + 1)
                }
            }
        }

        var length = 0
        for l in dp {
            length = max(l, length)
        }

        return length
    }
}
```
### Binary Search
- time complexity: O(nlogn)
- space complexity: O(n)
```swift
class Solution {
    func maxEnvelopes(_ envelopes: [[Int]]) -> Int {
        let n = envelopes.count
        let sortedEnvelopes = envelopes.sorted {
            return $0[0] == $1[0] ? $0[1] > $1[1] : $0[0] < $1[0]
        }
        var heights = [Int]()
        for envelope in sortedEnvelopes {
            heights.append(envelope[1]);
        }
        return lengthOfLIS(heights)
    }

    func lengthOfLIS(_ nums: [Int]) -> Int {
        var piles = 0, n = nums.count
        var top = Array(repeating: 0, count: n)
        for i in 0..<n {
            let poker = nums[i]
            var left = 0, right = piles
            while left < right {
                let mid = (left + right) / 2
                if top[mid] >= poker {
                    right = mid
                } else {
                    left = mid + 1
                }
            }
            if left == piles {
                piles += 1
            }
            top[left] = poker
        }
        return piles
    }
}
```
## 518. Coin Change 2
### DP
- time complexity: O(n\*amount)
- space complexity: O(n\*amount)
```swift
class Solution {
    func change(_ amount: Int, _ coins: [Int]) -> Int {
        if amount == 0 {
            return 1
        }
        if coins.isEmpty {
            return 0
        }
        let n = coins.count
        var dp = Array(repeating: Array(repeating: 0, count: amount + 1), count: n + 1)
        for i in 0...n {
            dp[i][0] = 1
        }

        for i in 1...n {
            for j in 1...amount {
                if j - coins[i - 1] < 0 {
                    dp[i][j] = dp[i - 1][j]
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]]
                }
            }
        }

        return dp[n][amount]
    }
}
```
### DP + Compression
- time complexity: O(n\*amount)
- space complexity: O(amount)
```swift
class Solution {
    func change(_ amount: Int, _ coins: [Int]) -> Int {
        if amount == 0 {
            return 1
        }
        if coins.isEmpty {
            return 0
        }
        let n = coins.count
        var dp = Array(repeating: 0, count: amount + 1)
        dp[0] = 1

        for i in 0..<n {
            for j in 1...amount {
                if j - coins[i] < 0 {
                    dp[j] = dp[j]
                } else {
                    dp[j] = dp[j] + dp[j - coins[i]]
                }
            }
        }

        return dp[amount]
    }
}
```
## 416. Partition Equal Subset Sum
### DP
- time complexity: O(n\*sum)
- space complexity: O(n\*sum)
```swift
class Solution {
    func canPartition(_ nums: [Int]) -> Bool {
        if nums.count == 1 {
            return false
        }

        var sum = 0
        for n in nums {
            sum += n
        }
        if sum % 2 == 1 {
            return false
        }
        
        let n = nums.count, s = sum / 2
        var dp = Array(repeating: Array(repeating: false, count: s + 1), count: n + 1)
        for i in 0..<n {
            dp[i][0] = true
        }
        for i in 1...n {
            for j in 1...s {
                if j - nums[i - 1] < 0 {
                    dp[i][j] = dp[i - 1][j]
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]]
                }
            }
        }
        return dp[n][s]
    }
}
```
### DP + Compression
- time complexity: O(n\*sum)
- space complexity: O(sum)
```swift
class Solution {
    func canPartition(_ nums: [Int]) -> Bool {
        if nums.count == 1 {
            return false
        }

        var sum = 0
        for n in nums {
            sum += n
        }
        if sum % 2 == 1 {
            return false
        }
        
        let n = nums.count, s = sum / 2
        var dp = Array(repeating: false, count: s + 1)
        dp[0] = true
        for i in 0..<n {
            for j in stride(from: s, through: 0, by: -1) {
                if j - nums[i] < 0 {
                    dp[j] = dp[j]
                } else {
                    dp[j] = dp[j] || dp[j - nums[i]]
                }
            }
        }
        return dp[s]
    }
}
```
## 53. Maximum Subarray
### DP
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func maxSubArray(_ nums: [Int]) -> Int {
        if nums.isEmpty {
            return 0
        }

        var n = nums.count
        var dp = Array(repeating: nums[0], count: n)
        for i in 1..<n {
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        }

        var ans = Int.min
        for i in 0..<n {
            ans = max(ans, dp[i])
        }

        return ans
    }
}
```
### DP + Compression
- time complexity: O(n)
- space complexity: O(1)
```swift
class Solution {
    func maxSubArray(_ nums: [Int]) -> Int {
        if nums.isEmpty {
            return 0
        }
        var n = nums.count
        var dp0 = nums[0], dp1 = nums[0], ans = nums[0]
        for i in 1..<n {
            dp1 = max(dp0 + nums[i], nums[i])
            dp0 = dp1
            ans = max(ans, dp1)
        }
        return ans
    }
}
```
## 494. Target Sum
### Backtracking
- time complexity: O(2^n), where n is length of noms array
- space complexity: O(n)
```swift
class Solution {
    func findTargetSumWays(_ nums: [Int], _ S: Int) -> Int {
        if nums.isEmpty {
            return 0
        }

        var results = 0

        func backtrack(_ pos: Int, _ leftSum: inout Int) {
            if pos == nums.count {
                if leftSum == 0 {
                    results += 1
                }
                return
            }

            for s in ["+", "-1"] {
                if s == "+" {
                    leftSum -= nums[pos]
                    backtrack(pos + 1, &leftSum)
                    leftSum += nums[pos]
                } else {
                    leftSum += nums[pos]
                    backtrack(pos + 1, &leftSum)
                    leftSum -= nums[pos]
                }
            }
        }

        var leftSum = S 
        backtrack(0, &leftSum)
        return results
    }
}
```
### Optimised with Memo
- time complexity: O(n\*sum)
- space complexity: O(n\*sum)
```swift
class Solution {
    func findTargetSumWays(_ nums: [Int], _ S: Int) -> Int {
        if nums.isEmpty {
            return 0
        }
        var memo = [String: Int]()

        func dp(_ pos: Int, _ leftSum: Int) -> Int {
            if pos == nums.count {
                if leftSum == 0 {
                    return 1
                }
                return 0
            }
            let key = "\(pos),\(leftSum)"
            if memo[key] != nil {
                return memo[key]!
            }

            memo[key] = dp(pos + 1, leftSum - nums[pos]) + dp(pos + 1, leftSum + nums[pos])
            return memo[key]!
        }

        return dp(0, S)
    }
}
```
### DP
- time complexity: O(nS)
- space complexity: O(nS)
```swift
class Solution {
    func findTargetSumWays(_ nums: [Int], _ S: Int) -> Int {
        if nums.isEmpty {
            return 0
        }
        var sum = 0
        for n in nums {
            sum += n
        }
        if sum < S || (sum + S) % 2 == 1 {
            return 0
        }
        return subsets(nums, (sum + S) / 2)
    }

    func subsets(_ nums: [Int], _ S: Int) -> Int {
        let n = nums.count
        var dp = Array(repeating: Array(repeating: 0, count: S + 1), count: n + 1)
        for i in 0...n {
            dp[i][0] = 1
        }
        for i in 1...n {
            for j in 0...S {
                if j >= nums[i - 1] {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]]
                } else {
                    dp[i][j] = dp[i - 1][j]
                }
            }
        }
        return dp[n][S]
    }
}
```
### DP + Compression
- time complexity: O(nS)
- space complexity: O(S)
```swift
class Solution {
    func findTargetSumWays(_ nums: [Int], _ S: Int) -> Int {
        if nums.isEmpty {
            return 0
        }
        var sum = 0
        for n in nums {
            sum += n
        }
        if sum < S || (sum + S) % 2 == 1 {
            return 0
        }
        return subsets(nums, (sum + S) / 2)
    }

    func subsets(_ nums: [Int], _ S: Int) -> Int {
        let n = nums.count
        var dp = Array(repeating: 0, count: S + 1)
        dp[0] = 1
        for i in 1...n {
            for j in stride(from: S, through: 0, by: -1) {
                if j >= nums[i - 1] {
                    dp[j] = dp[j] + dp[j - nums[i - 1]]
                } else {
                    dp[j] = dp[j]
                }
            }
        }
        return dp[S]
    }
}
```
## 1312. Minimum Insertion Steps to Make a String Palindrome
### DP
- time complexity: O(n^2)
- space complexity: O(n^2)
```swift
class Solution {
    func minInsertions(_ s: String) -> Int {
        if s.count == 1 {
            return 0
        }
        let sArr = [Character](s)
        let n = sArr.count
        var dp = Array(repeating: Array(repeating: 0, count: n), count: n)
        for i in stride(from: n - 2, through: 0, by: -1) {
            for j in i+1..<n {
                if sArr[i] == sArr[j] {
                    dp[i][j] = dp[i + 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i][j - 1], dp[i + 1][j]) + 1
                }
            }
        }
        return dp[0][n - 1]
    }
}
```
### DP with State Compression
- time complexity: O(n^2)
- space complexity: O(n)
```swift
class Solution {
    func minInsertions(_ s: String) -> Int {
        if s.count == 1 {
            return 0
        }
        let sArr = [Character](s)
        let n = sArr.count
        var dp = Array(repeating: 0, count: n)
        for i in stride(from: n - 2, through: 0, by: -1) {
            var pre = 0
            for j in i+1..<n {
                let tmp = dp[j]
                if sArr[i] == sArr[j] {
                    dp[j] = pre
                } else {
                    dp[j] = min(dp[j - 1], dp[j]) + 1
                }
                pre = tmp
            }
        }
        return dp[n - 1]
    }
}
```
## 496. Next Greater Element I
- time complexity: O(m + n)
- space complexity: O(n), n is length of nums2
```swift
class Solution {
    func nextGreaterElement(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        var map = [Int: Int]()
        var stack = [Int]()
        var res = [Int]()
        for i in 0..<nums2.count {
            while !stack.isEmpty && nums2[i] > stack.last! {
                map[stack.removeLast()] = nums2[i]
            }
            stack.append(nums2[i])
        }
        while !stack.isEmpty {
            map[stack.removeLast()] = -1
        }
        for i in nums1 {
            res.append(map[i]!)
        }
        return res
    }
}
```
## 682. Baseball Game
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func calPoints(_ ops: [String]) -> Int {
        var stack = [Int]()
        var scores = 0
        for i in ops {
            if i == "C" {
                let val = stack.removeLast()
                scores -= val
            } else if i == "D" {
                let val = 2 * stack.last!
                scores += val
                stack.append(val)
            } else if i == "+" {
                let top = stack.removeLast()
                let val = top + stack.last!
                scores += val
                stack.append(top)
                stack.append(val)
            } else {
                let val = Int(i)!
                scores += val
                stack.append(val)
            }
        }
        return scores
    }
}
```
## 844. Backspace String Compare
### Rebuild String
- time complexity: O(m + n)
- space complexity: O(m + n)
```swift
class Solution {
    func backspaceCompare(_ S: String, _ T: String) -> Bool {
        return build(S) == build(T)
    }

    func build(_ str: String) -> String {
        var s = [Character]()
        for i in str {
            if i != "#" {
                s.append(i)
            } else {
                s.popLast()
            }
        }
        return String(s)
    }
}
```
### Two Pointer
- time complexity: O(m + n)
- space complexity: O(1)
```swift
class Solution {
    func backspaceCompare(_ S: String, _ T: String) -> Bool {
        var sArray = [Character](S), tArray = [Character](T)
        var i = sArray.count - 1, j = tArray.count - 1
        var skipS = 0, skipT = 0
        while i >= 0 || j >= 0 {
            while i >= 0 {
                if sArray[i] == "#" {
                    skipS += 1
                    i -= 1
                } else if skipS > 0 {
                    i -= 1
                    skipS -= 1
                } else {
                    break
                }
            }

            while j >= 0 {
                if tArray[j] == "#" {
                    skipT += 1
                    j -= 1
                } else if skipT > 0 {
                    j -= 1
                    skipT -= 1
                } else {
                    break
                }
            }

            if i >= 0 && j >= 0 && sArray[i] != tArray[j] {
                return false
            }

            if (i >= 0) != (j >= 0) {
                return false
            }

            i -= 1
            j -= 1
        }

        return true
    }
}
```
## 20. Valid Parentheses
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func isValid(_ s: String) -> Bool {
        if s.count % 2 == 1 {
            return false
        }
        var stack = [Character]()
        var dict: [Character: Character] = [")": "(", "}": "{", "]": "["]
        for c in s {
            if c == "(" || c == "{" || c == "[" {
                stack.append(c)
            } else {
                if !stack.isEmpty && stack.last! == dict[c]! {
                    stack.popLast()
                } else {
                    return false
                }
            }
        }

        return stack.isEmpty
    }
}
```
## 876. Middle of the Linked List
### Use Array
- time complexity: O(n)
- space complexity: O(n)
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
    func middleNode(_ head: ListNode?) -> ListNode? {
        var arr = [ListNode?]()
        var cur = head
        while cur != nil {
            arr.append(cur)
            cur = cur?.next
        }
        return arr[arr.count / 2]
    }
}
```
### Two Pass
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
    func middleNode(_ head: ListNode?) -> ListNode? {
        var count = 0
        var cur = head
        while cur != nil {
            count += 1
            cur = cur?.next
        }
        cur = head
        var i = 0
        while i < count / 2 {
            cur = cur?.next
            i += 1
        }
        return cur
    }
}
```
### Two Pointer
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
    func middleNode(_ head: ListNode?) -> ListNode? {
        var slow = head, fast = head?.next
        while fast != nil {
            slow = slow?.next
            fast = fast?.next?.next
        } 
        return slow
    }
}
```
## 1. Two Sum
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
        var dict = [Int: Int]()
        for i in 0..<nums.count {
            if dict[target - nums[i]] != nil {
                return [dict[target - nums[i]]!, i]
            }
            dict[nums[i]] = i
        }
        return [-1, -1]
    }
}
```
## 114. Flatten Binary Tree to Linked List
- time complexity: O(n)
- space complexity: O(n)
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
    func flatten(_ root: TreeNode?) {
        if root == nil {
            return
        }

        flatten(root?.left)
        flatten(root?.right)

        let left = root?.left
        let right = root?.right

        // move left subtree to right side
        root?.right = left
        root?.left = nil

        // connect flatten right subtree to the end of left subtree
        var cur = root
        while cur?.right != nil {
            cur = cur?.right
        }
        cur?.right = right
    }
}
```
## 226. Invert Binary Tree
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
    func invertTree(_ root: TreeNode?) -> TreeNode? {
        if root == nil {
            return nil
        }

        var temp = root?.left
        root?.left = root?.right
        root?.right = temp

        invertTree(root?.left)
        invertTree(root?.right)

        return root
    }
}
```
## 93. Restore IP Addresses
- time complexity: O(3^subStringCount)
- space complexity: O(subStringCount)
```swift
class Solution {
    func restoreIpAddresses(_ s: String) -> [String] {
        let sArray = [Character](s)
        let n = sArray.count
        var res = [String]()

        func backtrack(_ start: Int, _ tempList: inout [String]) {
            if start == n && tempList.count == 4 {
                res.append(tempList.joined(separator: "."))
                return
            }

            if tempList.count == 4 && start <= n {
                return
            }

            for i in 0..<3 {
                if start + i >= sArray.count || (i != 0 && sArray[start] == "0") {
                    continue
                }
                let subString = String(sArray[start...start+i])
                if Int(subString)! > 255 {
                    continue
                }
                tempList.append(subString)
                backtrack(start + i + 1, &tempList)
                tempList.popLast()
            }
        }

        var tempList = [String]()
        backtrack(0, &tempList)
        return res
    }
}
```
## 5. Longest Palindromic Substring
- time complexity: O(n^2)
- space complexity: O(n^2)
```swift
class Solution {
    func longestPalindrome(_ s: String) -> String {
        var sArray = [Character](s)
        if sArray.isEmpty {
            return ""
        }

        var begin = 0, maxLength = 0
        let n = sArray.count
        var dp = Array(repeating: Array(repeating: false, count: n), count: n)
        for right in 0..<n {
            for left in 0...right {
                if sArray[left] == sArray[right] && (right - left <= 2 || dp[left + 1][right - 1]) {
                    dp[left][right] = true
                    if right - left + 1 > maxLength {
                        maxLength = right - left + 1
                        begin = left
                    }
                }
            }
        }
        return String(sArray[begin..<begin+maxLength])
    }
}
```
## 131. Palindrome Partitioning
### Backtrack
- time complexity: O(n\*2^n)
- space complexity: O(n)
```swift
class Solution {
    func partition(_ s: String) -> [[String]] {
        var sArray = [Character](s)
        var res = [[String]]()
        if sArray.isEmpty {
            return res
        }

        func backtrack(_ start: Int, _ tempList: inout [String]) {
            if start == sArray.count {
                res.append(tempList)
                return
            }

            for i in start..<sArray.count {
                if !checkPalindrome(start, i) {
                    continue
                }
                tempList.append(String(sArray[start...i]))
                backtrack(i + 1, &tempList)
                tempList.popLast()
            }
        }

        func checkPalindrome(_ left: Int, _ right: Int) -> Bool {
            var left = left, right = right
            while left < right {
                if sArray[left] != sArray[right] {
                    return false
                }
                left += 1
                right -= 1
            }
            return true
        }

        var tempList = [String]()
        backtrack(0, &tempList)
        return res
    }
}
```
### Optimised Backtrack with DP
- time complexity: O(2^n)
- space complexity: O(n^2)
```swift
class Solution {
    func partition(_ s: String) -> [[String]] {
        var sArray = [Character](s)
        var res = [[String]]()
        let n = sArray.count
        if sArray.isEmpty {
            return res
        }
        var dp = Array(repeating: Array(repeating: false, count: n), count: n)

        func backtrack(_ start: Int, _ tempList: inout [String]) {
            if start == sArray.count {
                res.append(tempList)
                return
            }

            for i in start..<sArray.count {
                if !dp[start][i] {
                    continue
                }
                tempList.append(String(sArray[start...i]))
                backtrack(i + 1, &tempList)
                tempList.popLast()
            }
        }

        func fillInDPTable(){
            for right in 0..<n {
                for left in 0...right {
                    if sArray[left] == sArray[right] && (right - left <= 2 || dp[left + 1][right - 1]) {
                        dp[left][right] = true
                    }
                }
            }
        }

        fillInDPTable()
        var tempList = [String]()
        backtrack(0, &tempList)
        return res
    }
}
```
## 17. Letter Combinations of a Phone Number
- time complexity: O(3^m\*4^n), where m is the number of three letters digits and n is the number of four letters digits
- space complexity: O(m + n)
```swift
class Solution {
    func letterCombinations(_ digits: String) -> [String] {
        if digits.count == 0 {
            return []
        }
        var digitArray = [Character](digits)
        var dict: [Character: [Character]] = ["2": ["a", "b", "c"], "3": ["d", "e", "f"], "4": ["g", "h", "i"], "5": ["j", "k", "l"],
        "6": ["m", "n", "o"], "7": ["p", "q", "r", "s"], "8": ["t", "u", "v"], "9": ["w", "x", "y", "z"]]
        var res = [String]()

        func backtrack(_ pos: Int, _ tempList: inout [Character]) {
            if pos == digitArray.count {
                res.append(String(tempList))
                return
            }
            let digit = digitArray[pos]
            let letters = dict[digit]!
            for i in 0..<letters.count {
                tempList.append(letters[i])
                backtrack(pos + 1, &tempList)
                tempList.popLast()
            }
        }

        var tempList = [Character]()
        backtrack(0, &tempList)
        return res
    }
}
```
## 39. Combination Sum
- time complexity: O(2^n)
- space complexity: O(target)
```swift
class Solution {
    func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
        if candidates.isEmpty {
            return []
        }

        var res = [[Int]]()
        var sortedNums = candidates.sorted()

        func backtrack(_ pos: Int, _ tempSum: Int, _ tempList: inout [Int]) {
            if tempSum == target {
                res.append(tempList)
                return
            }

            if tempSum > target {
                return
            }

            for i in pos..<sortedNums.count {
                tempList.append(sortedNums[i])
                backtrack(i, tempSum + sortedNums[i], &tempList)
                tempList.popLast()
            }
        }

        var tempList = [Int]()
        backtrack(0, 0, &tempList)
        return res
    }
}
```
## 47. Permutations II
- time complexity: O(n!)
- space complexity: O(n)
```swift
class Solution {
    func permuteUnique(_ nums: [Int]) -> [[Int]] {
        var res = [[Int]]()
        var sortedNums = nums.sorted()
        var visited = Array(repeating: false, count: nums.count)

        func backtrack(_ track: inout [Int]) {
            if track.count == sortedNums.count {
                res.append(track)
                return
            }

            for i in 0..<sortedNums.count {
                if visited[i] || (i > 0 && sortedNums[i] == sortedNums[i - 1] && !visited[i - 1]) {
                    continue
                }
                visited[i] = true
                track.append(sortedNums[i])
                backtrack(&track)
                visited[i] = false
                track.popLast()
            }
        }

        var track = [Int]()
        backtrack(&track)
        return res
    }
}
```
## 46. Permutations
- time complexity: O(n!)
- space complexity: O(n), without considering results space
```swift
class Solution {
    func permute(_ nums: [Int]) -> [[Int]] {
        var res = [[Int]]()
        var visited = Array(repeating: false, count: nums.count)

        func backtrack(_ track: inout [Int]) {
            if track.count == nums.count {
                res.append(track)
                return
            }

            for i in 0..<nums.count {
                if visited[i] {
                    continue
                }
                visited[i] = true
                track.append(nums[i])
                backtrack(&track)
                visited[i] = false
                track.popLast()
            }
        }

        var track = [Int]()
        backtrack(&track)
        return res
    }
}
```
## 90. Subsets II
- time complexity: O(2^n)
- space complexity: O(n)
```swift
class Solution {
    func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
        var res = [[Int]]()
        var sortedNums = nums.sorted()
        
        func backtrack(_ pos: Int, _ track: inout [Int]) {
            res.append(track)
            for i in pos..<sortedNums.count {
                if i > pos && sortedNums[i] == sortedNums[i - 1] {
                    continue
                }
                track.append(sortedNums[i])
                backtrack(i + 1, &track)
                track.popLast()
            }
        }
        
        var tempList = [Int]()
        backtrack(0, &tempList)
        return res
    }
}
```
## 78. Subsets
- time complexity: O(2^n)
- space complexity: O(n)
```swift
class Solution {
    func subsets(_ nums: [Int]) -> [[Int]] {
        var res = [[Int]]()
        
        func backtrack(_ pos: Int, _ track: inout [Int]) {
            res.append(track)
            for i in pos..<nums.count {
                track.append(nums[i])
                backtrack(i + 1, &track)
                track.popLast()
            }
        }
        
        var tempList = [Int]()
        backtrack(0, &tempList)
        return res
    } 
}
```
## 450. Delete Node in a BST
- time complexity: O(n)
- space complexity: O(n)
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
    func deleteNode(_ root: TreeNode?, _ key: Int) -> TreeNode? {        
        guard let root = root else {
            return nil/**
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
    func isBalanced(_ root: TreeNode?) -> Bool {
        if root == nil {
            return true
        }

        if abs(height(root?.left) - height(root?.right)) > 1 {
            return false
        }

        return isBalanced(root?.left) && isBalanced(root?.right)
    }

    func height(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }

        return max(height(root?.left), height(root?.right)) + 1
    }
}
        }
        
        if root.val > key {
            root.left = deleteNode(root.left, key)
        } else if root.val < key {
            root.right = deleteNode(root.right, key)
        } else {
            if root.left == nil {
                return root.right
            } else if root.right == nil {
                return root.left
            } else {
                var node = root.right
                while node?.left != nil {
                    node = node?.left
                }
                node?.left = root.left
                return root.right
            }
        }
        
        return root
    }
}
```
## 3. Longest Substring Without Repeating Characters
- time complexity: O(n)
- space complexity: O(K), K is number of distinct characters
```swift
class Solution {
    func lengthOfLongestSubstring(_ s: String) -> Int {
        var arr = [Character](s)
        var dict = [Character: Int]()
        var start = 0, maxLength = 0
        for end in 0..<arr.count {
            let rightChar = arr[end]
            if dict[rightChar] != nil {
                start = max(start, dict[rightChar]! + 1)
            }
            dict[rightChar] = end
            maxLength = max(maxLength, end - start + 1)
        }
        return maxLength
    }
}
```
## 438. Find All Anagrams in a String
- time complexity: O(s + p)
- space complexity: O(p)
```swift
class Solution {
    func findAnagrams(_ s: String, _ p: String) -> [Int] {
        var sArray = [Character](s)
        var map = [Character: Int]()
        for i in p {
            map[i, default: 0] += 1
        }
        var left = 0, matched = 0
        var results = [Int]()
        
        for right in 0..<sArray.count {
            let rightChar = sArray[right]
            if map[rightChar] != nil {
                map[rightChar]! -= 1
                if map[rightChar]! == 0 {
                    matched += 1
                }
            }

            while matched == map.count {
                if right - left + 1 == p.count {
                    results.append(left)
                }

                let leftChar = sArray[left]
                if map[leftChar] != nil {
                    if map[leftChar]! == 0 {
                        matched -= 1
                    }
                    map[leftChar]! += 1
                }

                left += 1
            }
        }

        return results
    }
}
```
## 567. Permutation in String
- time complexity: O(s1 + s2)
- space complexity: O(s1 + s2)
```swift
class Solution {
    func checkInclusion(_ s1: String, _ s2: String) -> Bool {
        var map = [Character: Int](), window = [Character: Int]()
        var s2Array = [Character](s2)
        var matched = 0
        for i in s1 {
            map[i, default: 0] += 1
        }
        
        var left = 0
        for right in 0..<s2Array.count {
            let rightChar = s2Array[right]
            if map[rightChar] != nil {
                window[rightChar, default: 0] += 1
                if window[rightChar] == map[rightChar] {
                    matched += 1
                }
            }
            
            while matched == map.count {
                if right - left + 1 == s1.count {
                    return true
                }
                let leftChar = s2Array[left]
                if map[leftChar] != nil {
                    if window[leftChar] == map[leftChar] {
                        matched -= 1
                    }
                    window[leftChar]! -= 1
                }
                
                left += 1
            }
        }
        
        return false
    }
}
```
## 76. Minimum Window Substring
- time complexity: O(s + t)
- space complexity: O(t)
```swift
class Solution {
    func minWindow(_ s: String, _ t: String) -> String {
        if s.count == 0 || t.count == 0 || s.count < t.count {
            return ""
        }
        
        var sArray = [Character](s), tArray = [Character](t)
        var map = [Character: Int]()
        var start = 0, missingTypes = 0, resL = 0, resR = 0, minLength = sArray.count + 1
        
        for i in tArray {
            if map[i] == nil {
                map[i] = 0
                missingTypes += 1
            }
            map[i]! += 1
        }
        
        for end in 0..<sArray.count {
            let rightChar = sArray[end]
            if map[rightChar] != nil {
                map[rightChar]! -= 1  
                if map[rightChar] == 0 {
                    missingTypes -= 1
            }
        }
            
        while missingTypes == 0 {
            if end - start + 1 < minLength {
                minLength = end - start + 1
                resL = start
                resR = end
            }
            let leftChar = sArray[start]
            if map[leftChar] != nil {
                map[leftChar]! += 1
                if map[leftChar]! > 0 {
                    missingTypes += 1
                    }
                }
            start += 1
            }
        }
        return minLength == sArray.count + 1 ? "" : String(sArray[resL...resR])

    }
}
```
## 509. Fibonacci Number
### Top Down + Memorisation
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    var cache = [Int: Int]()
    func fib(_ N: Int) -> Int {
        if N <= 1 {
            return N
        }

        if cache[N] != nil {
            return cache[N]!
        }

        cache[N] = fib(N - 1) + fib(N - 2)
        return cache[N]!
    }
}
```
### DP
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func fib(_ N: Int) -> Int {
        if N <= 1 {
            return N
        }
        
        var dp = Array(repeating: 0, count: N + 1)
        dp[1] = 1
        for i in 2...N {
            dp[i] = dp[i - 1] + dp[i - 2]
        }
        return dp[N]
    }
}
```
### DP with O(1) space
- time complexity: O(n)
- space complexity: O(1)
```swift
class Solution {
    func fib(_ N: Int) -> Int {
        if N <= 1 {
            return N
        }

        var dp1 = 0, dp2 = 1
        for i in 2...N {
            let temp = dp2
            dp2 += dp1
            dp1 = temp
        }
        return dp2
    }
}
```
## 95. Unique Binary Search Trees II
- time complexity: O(nGn), there are Gn BST, each takes O(n) time to construct
- space complexity: O(nGn)
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
    func generateTrees(_ n: Int) -> [TreeNode?] {
        if n == 0 {
            return []
        }
        return generateTrees(1, n)
    }
    
    func generateTrees(_ start: Int, _ end: Int) -> [TreeNode?] {
        if start > end {
            return [nil]
        }
        var allTrees = [TreeNode?]()
        
        for i in start...end {
            let leftTrees = generateTrees(start, i - 1)
            let rightTrees = generateTrees(i + 1, end)
            
            for l in leftTrees {
                for r in rightTrees {
                    let currentTree = TreeNode(i)
                    currentTree.left = l
                    currentTree.right = r
                    allTrees.append(currentTree)
                }
            }
        }
        
        return allTrees
    }
}
```
## 24. Swap Nodes in Pairs
### Recursion
- time complexity: O(n)
- space complexity: O(n)
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
    func swapPairs(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }

        let firstNode = head, secondNode = head?.next, nextHead = secondNode?.next
        secondNode?.next = firstNode
        firstNode?.next = swapPairs(nextHead)
        return secondNode
    }
}
```
### Iteration
- time complexity: O(n)
- space complexity: O(1)
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
    func swapPairs(_ head: ListNode?) -> ListNode? {
        let dummyHead: ListNode? = ListNode(-1)
        dummyHead?.next = head
        var prev = dummyHead
        var cur = head

        while cur != nil && cur?.next != nil {
            let firstNode = cur, secondNode = cur?.next
            firstNode?.next = secondNode?.next
            secondNode?.next = firstNode
            prev?.next = secondNode

            prev = firstNode
            cur = firstNode?.next
        }

        return dummyHead?.next
    }
}
```
## 344. Reverse String
### Recursion
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func reverseString(_ s: inout [Character]) {
        reverse(&s, 0, s.count - 1)
    }

    func reverse(_ s: inout [Character], _ left: Int, _ right: Int) {
        if left >= right {
            return
        }
        s.swapAt(left, right)
        reverse(&s, left + 1, right - 1)
    }
}
```
### Two Pointers
- time complexity: O(n)
- space complexity: O(1)
```swift
class Solution {
    func reverseString(_ s: inout [Character]) {
        var left = 0, right = s.count - 1
        while left < right {
            s.swapAt(left, right)
            left += 1
            right -= 1
        }
    }
}
```
## 92. Backpack (LintCode)
- time complexity: O(mn)
- space complexity: O(mn)
```swift
class Solution {
    func backPack(_ m: Int, _ A: [Int]) -> Int {
        let n = A.count
        var dp = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1) 
        for i in 1...n {
            for w in 1...m {
                if w - A[i - 1] < 0 {
                    dp[i][w] = dp[i - 1][w]
                } else {
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - A[i - 1]] + A[i - 1])
                }
            }
        }
        
        return dp[n][m]
    }
}
```
## 125. Backpack II (LintCode)
- time complexity: O(mn)
- space complexity: O(mn)
```swift
class Solution {
    func backPackII(_ m: Int, _ weight: [Int], _ value: [Int]) -> Int {
        let n = weight.count
        var dp = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1)
        for i in 1...n {
            for w in 1...m {
                if w - weight[i - 1] < 0 {
                    dp[i][w] = dp[i - 1][w]
                } else {
                    dp[i][w] = max(dp[i - 1][w - weight[i - 1]] + value[i - 1], dp[i - 1][w])
                }
            }
        }
        return dp[n][m]
    }
}
```
## 322. Coin Change
### Top Down + Memorisation
- time complexity: O(nk), k is number of denominations
- space complexity: O(n)
```swift
class Solution {
    var dict = [Int: Int]()
    
    func coinChange(_ coins: [Int], _ amount: Int) -> Int {
        return dp(coins, amount)
    }
    
    func dp(_ coins: [Int], _ amount: Int) -> Int {
        if amount == 0 {
            return 0
        }
        if amount < 0 {
            return -1
        }
        
        if dict[amount] != nil {
            return dict[amount]!
        }
        
        var res = Int.max
        for c in coins {
            var sub = dp(coins, amount - c)
            if sub == -1 {
                continue
            }
            res = min(res, sub + 1)
        }
        
        dict[amount] = res == Int.max ? -1 : res
        return dict[amount]!
    }
}
```
### Bottom Up
- time complexity: O(nk)
- space complexity: O(n)
```swift
class Solution {
    
    func coinChange(_ coins: [Int], _ amount: Int) -> Int {
        var dp = Array(repeating: amount + 1, count: amount + 1)
        dp[0] = 0
        for i in 1 ..< dp.count {
            for coin in coins {
                if i - coin < 0 {
                    continue
                }
                dp[i] = min(dp[i], 1 + dp[i - coin])
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount]
    }
}
```
## 72. Edit Distance
### Recursion
- time complexity: O(3^(l1+l2))
- space complexity: O(l1+l2)
```swift
class Solution {
    func minDistance(_ word1: String, _ word2: String) -> Int {
        if word1.isEmpty {
            return word2.count
        }
        if word2.isEmpty {
            return word1.count
        }

        let arr1 = [Character](word1)
        let arr2 = [Character](word2)

        func dp(_ i: Int, _ j: Int) -> Int {
            if i < 0 {
                return j + 1
            }
            if j < 0 {
                return i + 1
            }

            if arr1[i] == arr2[j] {
                return dp(i - 1, j - 1)
            } else {
                return min(dp(i, j - 1) + 1, dp(i - 1, j) + 1, dp(i - 1, j - 1) + 1)
            }
        }

        return dp(arr1.count - 1, arr2.count - 1)
    }
}
```
### Recursion + Memo
- time complexity: O(l1\*l2)
- space complexity: O(l1\*l2)
```swift
class Solution {
    func minDistance(_ word1: String, _ word2: String) -> Int {
        if word1.isEmpty {
            return word2.count
        }
        if word2.isEmpty {
            return word1.count
        }

        let arr1 = [Character](word1)
        let arr2 = [Character](word2)
        var memo = [String: Int]()

        func dp(_ i: Int, _ j: Int) -> Int {
            let key = "\(i),\(j)"
            if memo[key] != nil {
                return memo[key]!
            }
            if i < 0 {
                return j + 1
            }
            if j < 0 {
                return i + 1
            }

            if arr1[i] == arr2[j] {
                memo[key] = dp(i - 1, j - 1)
                return memo[key]!
            } else {
                memo[key] = min(dp(i, j - 1) + 1, dp(i - 1, j) + 1, dp(i - 1, j - 1) + 1)
                return memo[key]!
            }
        }

        return dp(arr1.count - 1, arr2.count - 1)
    }
}
```
### DP
- time complexity: O(l1\*l2)
- space complexity: O(l1\*l2)
```swift
class Solution {
    func minDistance(_ word1: String, _ word2: String) -> Int {
        if word1.isEmpty {
            return word2.count
        }
        if word2.isEmpty {
            return word1.count
        }

        let arr1 = [Character](word1), arr2 = [Character](word2)
        let l1 = arr1.count, l2 = arr2.count
        var dp = Array(repeating: Array(repeating: 0, count: l2 + 1), count: l1 + 1)
        for i in 1...l1 {
            dp[i][0] = i
        }
        for j in 1...l2 {
            dp[0][j] = j
        }
        
        for i in 1...l1 {
            for j in 1...l2 {
                if arr1[i - 1] == arr2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i][j - 1] + 1, dp[i - 1][j] + 1)
                }
            }
        }
        return dp[l1][l2]
    }
}
```
### DP + Compression
- time complexity: O(l1\*l2)
- space complexity: O(l2)
```swift
class Solution {
    func minDistance(_ word1: String, _ word2: String) -> Int {
        if word1.isEmpty {
            return word2.count
        }
        if word2.isEmpty {
            return word1.count
        }

        let arr1 = [Character](word1), arr2 = [Character](word2)
        let l1 = arr1.count, l2 = arr2.count
        var dp = Array(repeating: 0, count: l2 + 1)
        for j in 1...l2 {
            dp[j] = j
        }
        
        for i in 1...l1 {
            var pre = dp[0]
            dp[0] = i
            for j in 1...l2 {
                let tmp = dp[j]
                if arr1[i - 1] == arr2[j - 1] {
                    dp[j] = pre
                } else {
                    dp[j] = min(pre + 1, dp[j - 1] + 1, dp[j] + 1)
                }
                pre = tmp
            }
        }
        return dp[l2]
    }
}
```
## 1143. Longest Common Subsequence
### Recursion
- time complexity: O(2^(l1\*l2)) 
- space complexity: O(l1 + l2)
```swift
class Solution {
    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        var arr1 = [Character](text1), arr2 = [Character](text2)
        let l1 = arr1.count, l2 = arr2.count

        func dp(_ i: Int, _ j: Int) -> Int {
            if i < 0 || j < 0 {
                return 0
            }
            if arr1[i] == arr2[j] {
                return dp(i - 1, j - 1) + 1
            } else {
                return max(dp(i - 1, j), dp(i, j - 1))
            }
        }

        return dp(l1 - 1, l2 - 1)
    }
}
```
### Recursion + Memo
- time complexity: O(l1\*l2) 
- space complexity: O(l1\*l2)
```swift
class Solution {
    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        var arr1 = [Character](text1), arr2 = [Character](text2)
        let l1 = arr1.count, l2 = arr2.count
        var memo = [String: Int]()

        func dp(_ i: Int, _ j: Int) -> Int {
            if i < 0 || j < 0 {
                return 0
            }

            let key = "\(i),\(j)"
            if memo[key] != nil {
                return memo[key]!
            }
            if arr1[i] == arr2[j] {
                memo[key] = dp(i - 1, j - 1) + 1
                return memo[key]!
            } else {
                memo[key] = max(dp(i - 1, j), dp(i, j - 1))
                return memo[key]!
            }
        }

        return dp(l1 - 1, l2 - 1)
    }
}
```
### DP
- time complexity: O(l1\*l2) 
- space complexity: O(l1\*l2)
```swift
class Solution {
    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        var arr1 = [Character](text1), arr2 = [Character](text2)
        let l1 = arr1.count, l2 = arr2.count
        var dp = Array(repeating: Array(repeating: 0, count: l2 + 1), count: l1 + 1)
        for i in 1...l1 {
            for j in 1...l2 {
                if arr1[i - 1] == arr2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }
        return dp[l1][l2]
    }
}
```
### DP + Compression
- time complexity: O(l1\*l2) 
- space complexity: O(l2)
```swift
class Solution {
    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        var arr1 = [Character](text1), arr2 = [Character](text2)
        let l1 = arr1.count, l2 = arr2.count
        var dp = Array(repeating: 0, count: l2 + 1)
        for i in 1...l1 {
            var pre = 0
            for j in 1...l2 {
                let tmp = dp[j]
                if arr1[i - 1] == arr2[j - 1] {
                    dp[j] = pre + 1
                } else {
                    dp[j] = max(dp[j], dp[j - 1])
                }
                pre = tmp
            }
        }
        return dp[l2]
    }
}
```
## 139. Word Break
### DP
- time complexity: O(n^2) 
- space complexity: O(n)
```swift
class Solution {
    func wordBreak(_ s: String, _ wordDict: [String]) -> Bool {
        let set = Set(wordDict)
        var dp = Array(repeating: false, count: s.count + 1)
        dp[0] = true
        for i in 1...s.count {
            for j in 0..<i {
                let start = s.index(s.startIndex, offsetBy: j)
                let end = s.index(s.startIndex, offsetBy: i)
                let subStr = String(s[start..<end])
                if dp[j] && set.contains(subStr) {
                    dp[i] = true
                    break
                }
            }
        }
        return dp.last!
    }
}
```
## 300. Longest Increasing Subsequence
### DP
- time complexity: O(n^2) 
- space complexity: O(n)
```swift
class Solution {
    func lengthOfLIS(_ nums: [Int]) -> Int {
        if nums.isEmpty {
            return 0
        }

        var n = nums.count
        var dp = Array(repeating: 1, count: n)
        for i in 0..<n {
            for j in 0..<i {
                if nums[j] < nums[i] {
                    dp[i] = max(dp[i], dp[j] + 1)
                }
            }
        }

        var maxLength = 0
        for i in 0..<n {
            maxLength = max(maxLength, dp[i])
        }

        return maxLength
    }
}
```
### Binary Search
- time complexity: O(nlogn) 
- space complexity: O(n)
```swift
class Solution {
    func lengthOfLIS(_ nums: [Int]) -> Int {
        if nums.count <= 1 {
            return nums.count
        }
        
        var d = [Int]()
        for n in nums {
            if d.isEmpty || n > d.last! {
                d.append(n)
            } else {
                var l = 0, r = d.count - 1, loc = r
                while l <= r {
                    let mid = (l + r) / 2
                    if d[mid] >= n {
                        r = mid - 1
                        loc = mid
                    } else {
                        l = mid + 1
                    }
                }
                d[loc] = n
            }
        }
        return d.count
    }
}
```
## 132. Palindrome Partitioning II
### DP
- time complexity: O(n^3) 
- space complexity: O(n) 
```swift
class Solution {
    func minCut(_ s: String) -> Int {
        var s = Array(s).map { String($0) }
        if s.count == 0 || s.count == 1 {
            return 0
        }
        
        var dp = Array(repeating: 0, count: s.count)
        for i in 0..<s.count {
            dp[i] = i
            if isPalindrome(s, 0, i) {
                dp[i] = 0
                continue
            }
            
            for j in 0..<i {
                if isPalindrome(s, j + 1, i) {
                    dp[i] = min(dp[i], dp[j] + 1)
                }
            }
        }
        
        return dp.last!
    }
    
    func isPalindrome(_ s: [String], _ i: Int, _ j: Int) -> Bool {
        var i = i, j = j
        while i < j {
            if s[i] != s[j] {
                return false
            }
            i += 1
            j -= 1
        }
        return true
    }
}
```
### Optimised DP
- time complexity: O(n^2) 
- space complexity: O(n^2) 
```swift
class Solution {
    func minCut(_ s: String) -> Int {
        var s = Array(s).map { String($0) }
        if s.count == 0 || s.count == 1 {
            return 0
        }
        
        var dp = Array(repeating: 0, count: s.count)
        
        var checkPalindrome = Array(repeating: Array(repeating: false, count: s.count), count: s.count)
        for right in 0..<s.count {
            for left in 0...right {
                if s[left] == s[right] && (right - left <= 2 || checkPalindrome[left + 1][right - 1]) {
                    checkPalindrome[left][right] = true
                }
            }
        }
        
        for i in 0..<s.count {
            dp[i] = i
            if checkPalindrome[0][i] {
                dp[i] = 0
                continue
            } 
            for j in 0..<i {
                if checkPalindrome[j + 1][i] {
                    dp[i] = min(dp[i], dp[j] + 1)
                }
            }
        }
        
        return dp.last!
    }
}
```
## 45. Jump Game II
### Greedy
- time complexity: O(n) 
- space complexity: O(1)
```swift
class Solution {
    func jump(_ nums: [Int]) -> Int {
        var end = 0
        var maxPosition = 0
        var steps = 0
        for i in 0 ..< nums.count - 1 {
            maxPosition = max(maxPosition, nums[i] + i)
            if i == end {
                end = maxPosition
                steps += 1
            }
        }
        return steps
    }
}
```
## 55. Jump Game
### Greedy
- time complexity: O(n) 
- space complexity: O(1) 
```swift
class Solution {
    func canJump(_ nums: [Int]) -> Bool {
        var rightmost = 0
        for i in 0..<nums.count {
            if i <= rightmost {
                rightmost = max(rightmost, nums[i] + i)
                if rightmost >= nums.count - 1 {
                    return true
                }
            }
        }
        return false
    }
}
```
## 70. Climbing Stairs
### Recursion + Memorisation
- time complexity: O(n) 
- space complexity: O(n) 
```swift
class Solution {
    var table = [Int: Int]()

    func climbStairs(_ n: Int) -> Int {
        if n == 1 || n == 0 {
            return 1
        }

        if table[n] != nil {
            return table[n]!
        }
        table[n] = climbStairs(n - 1) + climbStairs(n - 2)
        return table[n]!
    }
}
```
### DP with O(n) space
- time complexity: O(n) 
- space complexity: O(n) 
```swift
class Solution {
    func climbStairs(_ n: Int) -> Int {
        var arr = Array(repeating: 1, count: n + 1) 
        for i in stride(from: 2, to: n + 1, by: 1) {
            arr[i] = arr[i - 1] + arr[i - 2]
        }
        return arr[n]
    }
}
```
### DP with O(1) space
- time complexity: O(n) 
- space complexity: O(1) 
```swift
class Solution {
    func climbStairs(_ n: Int) -> Int {
        var s1 = 1, s2 = 1 
        for i in stride(from: 2, to: n + 1, by: 1) {
            let temp = s2
            s2 += s1
            s1 = temp
        }
        return s2
    }
}
```
## 63. Unique Paths II
### DP
- time complexity: O(mn) 
- space complexity: O(mn) 
```swift
class Solution {
    func uniquePathsWithObstacles(_ obstacleGrid: [[Int]]) -> Int {
        if obstacleGrid[0][0] == 1 {
            return 0
        }
        
        let m = obstacleGrid.count, n = obstacleGrid[0].count
        var dp = Array(repeating: Array(repeating: 1, count: n), count: m)
        
        for i in 1..<m {
            if obstacleGrid[i][0] == 1 || dp[i - 1][0] == 0 {
                dp[i][0] = 0
            }
        }
        
        for j in 1..<n {
            if obstacleGrid[0][j] == 1 || dp[0][j - 1] == 0 {
                dp[0][j] = 0
            }
        }
        
        for i in 1..<m {
            for j in 1..<n {
                dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i - 1][j] + dp[i][j - 1]
            }
        }
        return dp[m - 1][n - 1]
    }
}
```
### DP + O(n) space
- time complexity: O(mn) 
- space complexity: O(n)
```swift
class Solution {
    func uniquePathsWithObstacles(_ obstacleGrid: [[Int]]) -> Int {
        if obstacleGrid[0][0] == 1 {
            return 0
        }
        
        let m = obstacleGrid.count, n = obstacleGrid[0].count
        var dp = Array(repeating: 0, count: n)
        dp[0] = 1
        
        for i in 0..<m {
            for j in 0..<n {
                if obstacleGrid[i][j] == 1 {
                    dp[j] = 0
                } else {
                    if j - 1 >= 0 {
                        dp[j] += dp[j - 1]
                    }  
                }
            }
        }
        return dp[n - 1]
    }
}
```
### DP + O(1) space
- time complexity: O(mn) 
- space complexity: O(1) 
```swift
class Solution {
    func uniquePathsWithObstacles(_ obstacleGrid: [[Int]]) -> Int {
        var obstacleGrid = obstacleGrid

        if obstacleGrid[0][0] == 1 {
            return 0
        }
        
        obstacleGrid[0][0] = 1

        let m = obstacleGrid.count, n = obstacleGrid[0].count
        
        for i in 1..<m {
            obstacleGrid[i][0] = (obstacleGrid[i][0] == 0 && obstacleGrid[i - 1][0] == 1) ? 1 : 0
        }
        
        for j in 1..<n {
            obstacleGrid[0][j] = (obstacleGrid[0][j] == 0 && obstacleGrid[0][j - 1] == 1) ? 1 : 0
        }
        
        for i in 1..<m {
            for j in 1..<n {
                obstacleGrid[i][j] = obstacleGrid[i][j] == 1 ? 0 : obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]
            }
        }
        return obstacleGrid[m - 1][n - 1]
    }
}
```
## 62. Unique Paths
### DP
- time complexity: O(mn) 
- space complexity: O(mn) 
```swift
class Solution {
    func uniquePaths(_ m: Int, _ n: Int) -> Int {
        var dp = Array(repeating: Array(repeating: 1, count: n), count: m)
        for i in 1..<m {
            for j in 1..<n {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
            }
        }
        return dp[m - 1][n - 1]
    }
}
```
### DP with O(n) space
- time complexity: O(mn) 
- space complexity: O(n) 
```swift
class Solution {
    func uniquePaths(_ m: Int, _ n: Int) -> Int {
        var dp = Array(repeating: 1, count: n)
        for i in 1..<m {
            for j in 1..<n {
                dp[j] = dp[j] + dp[j - 1]
            }
        }
        return dp[n - 1]
    }
}
```
## 64. Minimum Path Sum
### DP
- time complexity: O(mn) 
- space complexity: O(mn) 
```swift
class Solution {
    func minPathSum(_ grid: [[Int]]) -> Int {
        var dp = Array(repeating: Array(repeating: 0, count: grid[0].count), count: grid.count)
        for i in stride(from: grid.count - 1, through: 0, by: -1) {
            for j in stride(from: grid[0].count - 1, through: 0, by: -1) {
                if i == grid.count - 1 && j != grid[0].count - 1 {
                    dp[i][j] = grid[i][j] + dp[i][j + 1]
                } else if i != grid.count - 1 && j == grid[0].count - 1 {
                    dp[i][j] = grid[i][j] + dp[i + 1][j]
                } else if i != grid.count - 1 && j != grid[0].count - 1 {
                    dp[i][j] = grid[i][j] + min(dp[i][j + 1], dp[i + 1][j])
                } else {
                    dp[i][j] = grid[i][j]
                }
            }
        }
        return dp[0][0]
    }
}
```
### DP with O(n) space
- time complexity: O(mn) 
- space complexity: O(n) 
```swift
class Solution {
    func minPathSum(_ grid: [[Int]]) -> Int {
        var dp = Array(repeating: 0, count: grid[0].count)
        for i in stride(from: grid.count - 1, through: 0, by: -1) {
            for j in stride(from: grid[0].count - 1, through: 0, by: -1) {
                if i == grid.count - 1 && j != grid[0].count - 1 {
                    dp[j] = grid[i][j] + dp[j + 1]
                } else if i != grid.count - 1 && j == grid[0].count - 1 {
                    dp[j] = grid[i][j] + dp[j]
                } else if i != grid.count - 1 && j != grid[0].count - 1 {
                    dp[j] = grid[i][j] + min(dp[j + 1], dp[j])
                } else {
                    dp[j] = grid[i][j]
                }
            }
        }
        return dp[0]
    }
}
```
### DP with O(1) space
- time complexity: O(mn) 
- space complexity: O(1) 
```swift
class Solution {
    func minPathSum(_ grid: [[Int]]) -> Int {
        var grid = grid
        for i in stride(from: grid.count - 1, through: 0, by: -1) {
            for j in stride(from: grid[0].count - 1, through: 0, by: -1) {
                if i == grid.count - 1 && j != grid[0].count - 1 {
                    grid[i][j] = grid[i][j] + grid[i][j + 1]
                } else if i != grid.count - 1 && j == grid[0].count - 1 {
                    grid[i][j] = grid[i][j] + grid[i + 1][j]
                } else if i != grid.count - 1 && j != grid[0].count - 1 {
                    grid[i][j] = grid[i][j] + min(grid[i][j + 1], grid[i + 1][j])
                } else {
                    grid[i][j] = grid[i][j]
                }
            }
        }
        return grid[0][0]
    }
}
```
## 120. Triangle
### Top-down with Memoization
- time complexity: O(n^2) 
- space complexity: O(n^2) 
```swift
class Solution {
    var memo = [[Int?]]()
    func minimumTotal(_ triangle: [[Int]]) -> Int {
        memo = Array(repeating: Array(repeating: nil, count: triangle.count), count: triangle.count)
        return dfs(triangle, 0, 0)
    }
    
    func dfs(_ triangle: [[Int]], _ i: Int, _ j: Int) -> Int {
        if i == triangle.count {
            return 0
        }
        
        if memo[i][j] != nil {
            return memo[i][j]!
        }
        
        memo[i][j] = min(dfs(triangle, i + 1, j), dfs(triangle, i + 1, j + 1)) + triangle[i][j]
        
        return memo[i][j]!
    }
}
```
### Bottom-up with Tabulation
- time complexity: O(n^2) 
- space complexity: O(n^2) 
```swift
class Solution {
    func minimumTotal(_ triangle: [[Int]]) -> Int {
        var memo = Array(repeating: Array(repeating: 0, count: triangle.count + 1), count: triangle.count + 1)
        for i in stride(from: triangle.count - 1, through: 0, by: -1) {
            for j in 0...i {
                memo[i][j] = min(memo[i + 1][j], memo[i + 1][j + 1]) + triangle[i][j]
            }
        }
        return memo[0][0]
    }
}
```
### Optimised Tabulation
- time complexity: O(n^2) 
- space complexity: O(n)
```swift
class Solution {
    func minimumTotal(_ triangle: [[Int]]) -> Int {
        var memo = Array(repeating: 0, count: triangle.count + 1)
        for i in stride(from: triangle.count - 1, through: 0, by: -1) {
            for j in 0...i {
                memo[j] = min(memo[j], memo[j + 1]) + triangle[i][j]
            }
        }
        return memo[0]
    }
}
```
## 912. Sort an Array
### Bubble Sort
- time complexity: O(n^2) 
- space complexity: O(1) 
- stable: yes
```swift
class Solution {
    func sortArray(_ nums: [Int]) -> [Int] {
        var arr = nums
        bubbleSort(&arr)
        return arr
    }
    
    func bubbleSort(_ arr: inout [Int]) {
        for i in 0 ..< arr.count - 1 {
            for j in 0 ..< arr.count - i - 1 {
                if arr[j] > arr[j + 1] {
                    arr.swapAt(j, j+1)
                }
            }
        }
    }
}
```
### Selection Sort
- time complexity: O(n^2) 
- space complexity: O(1) 
- stable: no
```swift
class Solution {
    func sortArray(_ nums: [Int]) -> [Int] {
        var arr = nums
        selectionSort(&arr)
        return arr
    }
    
    func selectionSort(_ arr: inout [Int]) {
        for i in 0..<arr.count - 1 {
            var minIndex = i
            for j in i..<arr.count {
                if arr[j] < arr[minIndex] {
                    minIndex = j
                }
            }
            arr.swapAt(i, minIndex)
        }
    }
}
```
### Insertion Sort
- time complexity: O(n^2) 
- space complexity: O(1) 
- stable: yes
```swift
class Solution {
    func sortArray(_ nums: [Int]) -> [Int] {
        var arr = nums
        insertionSort(&arr)
        return arr
    }
    
    func insertionSort(_ arr: inout [Int]) {
        for i in 1..<arr.count {
            var pre = i - 1
            let cur = arr[i]
            while pre >= 0 && arr[pre] > cur {
                arr[pre + 1] = arr[pre]
                pre -= 1
            }
            arr[pre + 1] = cur
        }
    }
}
```
### Quick Sort
- time complexity: O(nlogn) average, O(n^2) worst case when arrays are sorted in ascending or descending order
- space complexity: O(logn) average, O(n) worst case
- stable: no
```swift
class Solution {
    func sortArray(_ nums: [Int]) -> [Int] {
        var nums = nums
        quickSort(&nums, 0, nums.count - 1)
        return nums
    }

    func quickSort(_ nums: inout [Int], _ left: Int, _ right: Int) {
        if left >= right {
            return
        }
        let p = partition(&nums, left, right)
        quickSort(&nums, left, p - 1)
        quickSort(&nums, p + 1, right)
    }

    func partition(_ nums: inout [Int], _ left: Int, _ right: Int) -> Int {
        var pivot = nums[right]
        var i = left
        for j in left..<right {
            if nums[j] < pivot {
                nums.swapAt(i, j)
                i += 1
            }
        }
        nums.swapAt(i, right)
        return i
    }
}
```
### Merge Sort
- time complexity: O(nlogn)
- space complexity: O(n)
- stable: yes, it actually depends on the implementation. If we change the line `left[l] <= right[r] ` to `left[l] < right[r] `, then the algorithm will become unstable
```swift
class Solution {
    func sortArray(_ nums: [Int]) -> [Int] {
        return mergeSort(nums)
    }

    func mergeSort(_ arr: [Int]) -> [Int] {
        if arr.count <= 1 {
            return arr
        }

        let mid = arr.count / 2
        let left = mergeSort(Array(arr[..<mid]))
        let right = mergeSort(Array(arr[mid...]))
        return merge(left, right)
    }

    func merge(_ left: [Int], _ right: [Int]) -> [Int] {
        var res = [Int]()
        var l = 0, r = 0
        while l < left.count && r < right.count {
            if left[l] <= right[r] {
                res.append(left[l])
                l += 1
            } else {
                res.append(right[r])
                r += 1
            }
        }
        if l < left.count {
            res.append(contentsOf: left[l...])
        }
        if r < right.count {
            res.append(contentsOf: right[r...])
        }
        return res
    }
}
```
### Heap Sort
- time complexity: O(nlogn)
- space complexity: O(1)
- stable: no
```swift
class Solution {
    func sortArray(_ nums: [Int]) -> [Int] {
        var arr = nums
        heapSort(&arr)
        return arr
    }
    
    func heapSort(_ arr: inout [Int]) {
        var length = arr.count
        buildMaxHeap(&arr, length)
        var l = arr.count
        for i in stride(from: arr.count - 1, through: 0, by: -1) {
            arr.swapAt(0, i)
            length -= 1
            heapify(&arr, 0, length)
        }
    }
    
    func buildMaxHeap(_ arr: inout [Int], _ length: Int) {
        for i in stride(from: length / 2, through: 0, by: -1) {
            heapify(&arr, i, length)
        }
    }
    
    func heapify(_ arr: inout [Int], _ i: Int, _ length: Int) {
        var left = 2 * i + 1, right = 2 * i + 2, largest = i
        if left < length && arr[left] > arr[largest] {
            largest = left
        }
        if right < length && arr[right] > arr[largest] {
            largest = right
        }
        if largest != i {
            arr.swapAt(i, largest)
            heapify(&arr, largest, length)
        }
    }
}
```
## 81. Search in Rotated Sorted Array II
- time complexity: O(logn) best case, O(n) worst case when all elements are identical
- space complexity: O(1)
```swift
class Solution {
    func search(_ nums: [Int], _ target: Int) -> Bool {
        var left = 0, right = nums.count - 1
        while left <= right {
            let mid = (left + right) / 2
            if nums[mid] == target {
                return true
            }
            if nums[left] == nums[mid] {
                left += 1
                continue
            }
            if nums[left] < nums[mid] {
                if nums[left] <= target && target < nums[mid] {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            } else {
                if target > nums[mid] && target <= nums[right] {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
        }
        return false
    }
}
```
## 33. Search in Rotated Sorted Array
- time complexity: O(logn)
- space complexity: O(1)
```swift
class Solution {
    func search(_ nums: [Int], _ target: Int) -> Int {
        var left = 0, right = nums.count - 1
        while left <= right {
            let mid = (left + right) / 2
            if nums[mid] == target {
                return mid
            } else if nums[mid] >= nums[left] {
                if target >= nums[left] && target < nums[mid] {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            } else {
                if target <= nums[right] && target > nums[mid] {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
        }
        return -1
    }
}
```
## 154. Find Minimum in Rotated Sorted Array II
- time complexity: O(logn) average, O(N) worst case if nums have all same elements
- space complexity: O(1)
```swift
class Solution {
    func findMin(_ nums: [Int]) -> Int {
        var left = 0, right = nums.count - 1
        while left <= right {
            let mid = (left + right) / 2
            if nums[mid] < nums[right] {
                right = mid
            } else if nums[mid] > nums[right] {
                left = mid + 1
            } else {
                right -= 1
            }
        }
        return nums[left]
    }
}
```
## 153. Find Minimum in Rotated Sorted Array
- time complexity: O(logn)
- space complexity: O(1)
```swift
class Solution {
    func findMin(_ nums: [Int]) -> Int {
        var left = 0, right = nums.count - 1
        while left < right {
            let mid = (left + right) / 2
            if nums[mid] < nums[right] {
                right = mid
            } else {
                left = mid + 1
            }
        }
        return nums[left]
    }
}
```
## 278. First Bad Version
- time complexity: O(logn)
- space complexity: O(1)
```swift
/**
 * The knows API is defined in the parent class VersionControl.
 *     func isBadVersion(_ version: Int) -> Bool{}
 */

class Solution : VersionControl {
    func firstBadVersion(_ n: Int) -> Int {
        var left = 1, right = n
        while left < right {
            let mid = (left + right) / 2
            if isBadVersion(mid) {
                right = mid
            } else {
                left = mid + 1
            }
        }
        return left
    }
}
```
## 74. Search a 2D Matrix
- time complexity: O(logn)
- space complexity: O(1)
```swift
class Solution {
    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
        if matrix.isEmpty {
            return false
        }
        
        let m = matrix.count, n = matrix[0].count
        var left = 0, right = m * n - 1
        while left <= right {
            let mid = (left + right) / 2
            let element = matrix[mid / n][mid % n]
            if element == target {
                return true
            } else if element < target {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return false
    }
}
```
## 61. Search for a Range (LintCode)
- time complexity: O(logn)
- space complexity: O(1)
```swift
class Solution {
    func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        if nums.isEmpty {
            return [-1, -1]
        }
        
        var res = [-1, -1]
        // find first occurence of target
        var left = 0, right = nums.count - 1
        while left + 1 < right {
            let mid = (left + right) / 2
            if nums[mid] > target {
                right = mid
            } else if nums[mid] < target {
                left = mid
            } else {
                right = mid
            }
        }
        
        if nums[left] == target {
            res[0] = left
        } else if nums[right] == target {
            res[0] = right
        } else {
            return res
        }
        
        // find second occurence of target
        left = 0
        right = nums.count - 1
        while left + 1 < right {
            let mid = (left + right) / 2
            if nums[mid] > target {
                right = mid
            } else if nums[mid] < target {
                left = mid
            } else {
                left = mid
            }
        }
        
        if nums[left] == target {
            res[1] = left
        } else if nums[right] == target {
            res[1] = right
        } else {
            return res
        }
        
        return res
    }
}
```
## 35. Search Insert Position
- time complexity: O(logn)
- space complexity: O(1)
```swift
class Solution {
    func searchInsert(_ nums: [Int], _ target: Int) -> Int {
        var left = 0, right = nums.count - 1
        while left <= right {
            let mid = (left + right) / 2
            if nums[mid] == target {
                return mid
            } else if nums[mid] < target {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return left
    }
}
```
## 704. Binary Search
- time complexity: O(logn)
- space complexity: O(1)
```swift
class Solution {
    func search(_ nums: [Int], _ target: Int) -> Int {
        var left = 0, right = nums.count - 1
        while left <= right {
            let mid = (left + right) / 2
            if nums[mid] == target {
                return mid
            } else if nums[mid] < target {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return -1
    }
}
```
## 201. Bitwise AND of Numbers Range
### Bit Shift
- time complexity: O(1)
- space complexity: O(1)
```swift
class Solution {
    func rangeBitwiseAnd(_ m: Int, _ n: Int) -> Int {
        var shift = 0
        var m = m, n = n
        while m < n {
            m >>= 1
            n >>= 1
            shift += 1
        }
        return m << shift
    }
}
```
### Turn-off rightmost 1 bits
- time complexity: O(1), require less iteration than above approach
- space complexity: O(1)
```swift
class Solution {
    func rangeBitwiseAnd(_ m: Int, _ n: Int) -> Int {
        var n = n
        while m < n {
            n &= n - 1
        }
        return m & n
    }
}
```
## 190. Reverse Bits
### Bit by bit reverse
- time complexity: O(1)
- space complexity: O(1)
```swift
class Solution {
    func reverseBits(_ n: Int) -> Int {
        var num = n
        var res = 0, power = 31
        while num != 0 {
            res += (num & 1) << power
            num >>= 1
            power -= 1
        }
        return res
    }
}
```
## 338. Counting Bits
### Pop Count
- time complexity: O(n\*k), k is number of bits in each number
- space complexity: O(n)
```swift
class Solution {
    func countBits(_ num: Int) -> [Int] {
        var res = [Int]()
        for n in 0...num {
            res.append(countOnes(n))
        }
        return res
    }

    func countOnes(_ n: Int) -> Int {
        var sum = 0
        var num = n
        while num != 0 {
            sum += 1
            num &= num - 1
        }
        return sum
    }
}
```
### DP + Most Significant Bit
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func countBits(_ num: Int) -> [Int] {
        var res = Array(repeating: 0, count: num + 1)
        var i = 0, b = 1
        while b <= num {
            // generate [b, 2b) or [b, num) from [0, b)
            while i < b && i + b <= num {
                res[i + b] = res[i] + 1
                i += 1
            }
            i = 0
            b = b << 1
        }
        return res
    }
}
```
### DP + Least Significant Bit
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func countBits(_ num: Int) -> [Int] {
        if num == 0 { return [0] }
        var res = Array(repeating: 0, count: num + 1)
        for i in 1...num {
            res[i] = res[i >> 1] + i & 1
        }
        return res
    }
}
```
### DP + Last Set Bit
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func countBits(_ num: Int) -> [Int] {
        if num == 0 { return [0] }
        var res = Array(repeating: 0, count: num + 1)
        for n in 1...num {
            res[n] = res[n & (n - 1)] + 1
        }
        return res
    }
}
```
## 191. Number of 1 Bits
### Loop and Flip
- time complexity: O(1)，cause integer is 32-bit, so always cost constant time
- space complexity: O(1)
```swift
class Solution {
    func hammingWeight(_ n: Int) -> Int {
        var bits = 0
        var mask = 1
        for _ in 0..<32 {
            if n & mask != 0 {
                bits += 1
            }
            mask = mask << 1
        }
        return bits
    }
}
```
### Bit Manipulation
- time complexity: O(1)
- space complexity: O(1)
```swift
class Solution {
    func hammingWeight(_ n: Int) -> Int {
        var sum = 0
        var num = n
        while num != 0 {
            sum += 1
            num &= num - 1
        }
        return sum
    }
}
```
## 260. Single Number III
### Hash Table
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> [Int] {
        var dict = [Int: Int]()
        var res = [Int]()
        for n in nums {
            if dict[n] == nil {
                dict[n] = 0
            }
            dict[n]! += 1
        }
        for (val, freq) in dict {
            if freq == 1 {
                res.append(val)
            }
        }
        return res
    }
}
```
### Bit-wise
- time complexity: O(n)
- space complexity: O(1)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> [Int] {
        // difference between two numbers which where seen only once
        var bitmask = 0
        for n in nums {
            bitmask ^= n
        }
        
        // rightmost 1-bit diff between x and y
        var diff = bitmask & (-bitmask)
        
        var x = 0
        for n in nums {
            if n & diff != 0 {
                x ^= n
            }
        }
        return [x, bitmask ^ x]
    }
}
```
## 137. Single Number II
### Hash Table
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        var dict = [Int: Int]()
        for n in nums {
            if dict[n] == nil {
                dict[n] = 0
            }
            dict[n]! += 1
        }
        for (val, freq) in dict {
            if freq == 1 {
                return val
            }
        }
        return 0
    }
}
```
### Math
(3(a+b+c) - (a+a+a+b+b+b+c)) / 2 = c
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        var set = Set<Int>()
        var sumOfSet = 0, sumOfNums = 0
        for n in nums {
            if !set.contains(n) {
                set.insert(n)
                sumOfSet += n
            }
            sumOfNums += n
        }
        return (3 * sumOfSet - sumOfNums) / 2
    }
}
```
### Bit Wise Operator
- time complexity: O(n)
- space complexity: O(1)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        var seenOnce = 0, seenTwice = 0
        for n in nums {
            seenOnce = ~seenTwice & (seenOnce ^ n)
            seenTwice = ~seenOnce & (seenTwice ^ n)
        }
        return seenOnce
    }
}
```
## 542. 01 Matrix
### BFS
- time complexity: O(m\*n), m is row number of matrix, n is column number of matrix
- space complexity: O(m\*n)
```swift
class Solution {
    func updateMatrix(_ matrix: [[Int]]) -> [[Int]] {
        var queue = [(x: Int, y: Int)]()
        var m = matrix
        let nr = matrix.count, nc = matrix[0].count
        var visited = Array(repeating: Array(repeating: false, count: nc), count: nr)
        
        for i in 0..<nr {
            for j in 0..<nc {
                if m[i][j] == 0 {
                    queue.append((i, j))
                    visited[i][j] = true
                }
            }
        }
        
        while !queue.isEmpty {
            let loc = queue.removeFirst()
            let val = m[loc.x][loc.y]
            for item: (dx: Int, dy: Int) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let x = loc.x + item.dx, y = loc.y + item.dy
                if x >= 0 && x < nr && y >= 0 && y < nc && visited[x][y] == false {
                    m[x][y] = val + 1
                    queue.append((x, y))
                    visited[x][y] = true
                }
                
            }
        }
        
        return m
    }
}
```
### DP
如果在一次遍历中比较四个方向，那么如果一个点自身不为0，位于角落且被三个同样不为0的点包围，遍有可能会得不到最小值，而是会卡在10000得不到最优解
- time complexity: O(m\*n)
- space complexity: O(m\*n)
```swift
class Solution {
    func updateMatrix(_ matrix: [[Int]]) -> [[Int]] {
        var m = matrix
        let nr = matrix.count, nc = matrix[0].count
        
        for i in 0..<nr {
            for j in 0..<nc {
                m[i][j] = m[i][j] == 0 ? 0 : 10000
            }
        }
        
        for i in 0..<nr {
            for j in 0..<nc {
                if i - 1 >= 0 {
                    m[i][j] = min(m[i][j], m[i - 1][j] + 1)
                }
                if j - 1 >= 0 {
                    m[i][j] = min(m[i][j], m[i][j - 1] + 1)
                }
            }
        }
        
        for i in (0..<nr).reversed() {
            for j in (0..<nc).reversed() {
                if i + 1 < nr {
                    m[i][j] = min(m[i][j], m[i + 1][j] + 1)
                }
                if j + 1 < nc {
                    m[i][j] = min(m[i][j], m[i][j + 1] + 1)
                }
            }
        }
        
        
        return m
    }
}
```
## 232. Implement Queue using Stacks
### Use two stacks-O(n) push, O(1) pop
- push: O(n)
- pop: O(1)
```swift
class MyQueue {

    /** Initialize your data structure here. */
    var stack1: [Int]
    var stack2: [Int]

    init() {
        stack1 = [Int]()
        stack2 = [Int]()    // store all queue elements
    }
    
    /** Push element x to the back of queue. */
    func push(_ x: Int) {
        while !stack2.isEmpty {
            stack1.append(stack2.removeLast())
        }
        stack2.append(x)
        while !stack1.isEmpty {
            stack2.append(stack1.removeLast())
        }
    }
    
    /** Removes the element from in front of queue and returns that element. */
    func pop() -> Int {
        return stack2.removeLast()
    }
    
    /** Get the front element. */
    func peek() -> Int {
        return stack2.last!
    }
    
    /** Returns whether the queue is empty. */
    func empty() -> Bool {
        return stack2.isEmpty
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * let obj = MyQueue()
 * obj.push(x)
 * let ret_2: Int = obj.pop()
 * let ret_3: Int = obj.peek()
 * let ret_4: Bool = obj.empty()
 */
```
### Two Stack - O(1) push, Amortized O(1) pop
- push: O(1)
- pop: Amortized O(1)
```swift
class MyQueue {

    /** Initialize your data structure here. */
    var stack1: [Int]
    var stack2: [Int]
    var front: Int?

    init() {
        stack1 = [Int]()
        stack2 = [Int]()
    }
    
    /** Push element x to the back of queue. */
    func push(_ x: Int) {
        if stack1.isEmpty {
            front = x
        }
        stack1.append(x)
    }
    
    /** Removes the element from in front of queue and returns that element. */
    func pop() -> Int {
        if stack2.isEmpty {
            while !stack1.isEmpty {
                stack2.append(stack1.removeLast())
            }
        }
        return stack2.removeLast()
    }
    
    /** Get the front element. */
    func peek() -> Int {
        if !stack2.isEmpty {
            return stack2.last!
        }
        return front!
    }
    
    /** Returns whether the queue is empty. */
    func empty() -> Bool {
        return stack1.isEmpty && stack2.isEmpty
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * let obj = MyQueue()
 * obj.push(x)
 * let ret_2: Int = obj.pop()
 * let ret_3: Int = obj.peek()
 * let ret_4: Bool = obj.empty()
 */ 
```
## 84. Largest Rectangle in Histogram
### Stack
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func largestRectangleArea(_ heights: [Int]) -> Int {
        var stack = [Int]()
        stack.append(-1)
        var maxArea = 0
        for i in 0..<heights.count {
            while stack.last! != -1 && heights[stack.last!] > heights[i] {
                maxArea = max(maxArea, heights[stack.removeLast()] * (i - stack.last! - 1))
            }
            stack.append(i)
        }
        while stack.last != -1 {
            maxArea = max(maxArea, heights[stack.removeLast()] * (heights.count - stack.last! - 1))
        }
        return maxArea
    }
}
```
## 200. Number of Islands
### DFS
- time complexity: O(M\*N), M is number of rows and N is number of columns
- space complexity: O(M\*N), worst case when the grid are filled with all lands
```swift
class Solution {

    func numIslands(_ grid: [[Character]]) -> Int {
        if grid.isEmpty || grid[0].isEmpty {
            return 0
        }
        
        var map = grid
        var num = 0
        let nr = map.count
        let nc = map[0].count
        
        for i in 0 ..< nr {
            for j in 0 ..< nc {
                if map[i][j] == "1" {
                    num += 1
                    dfs(&map, i, j)
                }
            }
        }
        
        return num
    }
    
    func dfs(_ map: inout [[Character]], _ r: Int, _ c: Int) {
        let nr = map.count
        let nc = map[0].count
        if r < 0 || c < 0 || r >= nr || c >= nc || map[r][c] == "0" {
            return
        }
         
        map[r][c] = "0"
        dfs(&map, r - 1, c)
        dfs(&map, r + 1, c)
        dfs(&map, r, c - 1)
        dfs(&map, r, c + 1)
    }
    
}
```
### BFS
- time complexity: O(M\*N)
- space complexity: O(min(M, N))
```swift
class Solution {
    func numIslands(_ grid: [[Character]]) -> Int {
        if grid.isEmpty || grid[0].isEmpty {
            return 0
        }
        
        var map = grid
        let nr = map.count, nc = map[0].count
        var num = 0
        for i in 0 ..< nr {
            for j in 0 ..< nc {
                if map[i][j] == "1" {
                    num += 1
                    map[i][j] = "0"
                    var queue = [(row: Int, col: Int)]()
                    queue.append((i, j))
                    
                    while !queue.isEmpty {
                        let loc = queue.removeFirst()
                        let row = loc.row, col = loc.col
                        if row - 1 >= 0 && map[row - 1][col] == "1" {
                            queue.append((row - 1, col))
                            map[row - 1][col] = "0"
                        }
                        if row + 1 < nr && map[row + 1][col] == "1" {
                            queue.append((row + 1, col))
108152902
                        }
                        if col - 1 >= 0 && map[row][col - 1] == "1" {
                            queue.append((row, col - 1))
                            map[row][col - 1] = "0"
                        }
                        if col + 1 < nc && map[row][col + 1] == "1" {
                            queue.append((row, col + 1))
                            map[row][col + 1] = "0"
                        }
                    }
                }
            }
        }
        
        return num
    }
}
```
### Union Find
- time complexity: O(M\*N)
- space complexity: O(M\*N)
```swift
class Solution {
    func numIslands(_ grid: [[Character]]) -> Int {
        if grid.isEmpty || grid[0].isEmpty {
            return 0
        }
        
        var map = grid
        let nr = map.count, nc = map[0].count
        let uf = UnionFind(map)
        for i in 0 ..< nr {
            for j in 0 ..< nc {
                if map[i][j] == "1" {
                    map[i][j] = "0"
                    if i + 1 < nr && map[i + 1][j] == "1" {
                        uf.union(i * nc + j, (i + 1) * nc + j)
                    }
                    if j + 1 < nc && map[i][j + 1] == "1" {
                        uf.union(i * nc + j, i * nc + j + 1)
                    }
                }
            }
        }
        return uf.count
    }
}

class UnionFind {
    var count: Int
    var parent: [Int]
    
    init(_ grid: [[Character]]) {
        count = 0
        let m = grid.count
        let n = grid[0].count
        parent = []
        for i in 0 ..< m {
            for j in 0 ..< n {
                parent.append(i * n + j)
                if grid[i][j] == "1" {
                    count += 1
                }
            }
        }
    }
    
    func find(_ i: Int) -> Int {
        var p = i
        while p != parent[p] {
            p = parent[p]
        }
        return p
    }
    
    func union(_ x: Int, _ y: Int) {
        let rootx = find(x)
        let rooty = find(y)
        if rootx != rooty {
            // raname rooty's component to rootx's name
            parent[rooty] = rootx
            count -= 1
        }
    }
}
```
## 133. Clone Graph
### DFS Search
- time complexity: O(n)
- space complexity: O(n)
```swift
/**
 * Definition for a Node.
 * public class Node {
 *     public var val: Int
 *     public var neighbors: [Node?]
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.neighbors = []
 *     }
 * }
 */

class Solution {
    var visited = [Int: Node?]()
    func cloneGraph(_ node: Node?) -> Node? {
        guard let cur = node else {
            return nil
        }

        if let item = visited[cur.val] {
            return item
        }

        let copyNode = Node(cur.val)
        visited[cur.val] = copyNode

        for n in cur.neighbors {
            copyNode.neighbors.append(cloneGraph(n))
        }

        return copyNode

    }
}
```
### BFS Search
- time complexity: O(n)
- space complexity: O(n)
```swift
/**
 * Definition for a Node.
 * public class Node {
 *     public var val: Int
 *     public var neighbors: [Node?]
 *     public init(_ val: Int) {
 *         self.val = val
 *         self.neighbors = []
 *     }
 * }
 */

class Solution {
    var visited = [Int: Node?]()

    func cloneGraph(_ node: Node?) -> Node? {
        if node == nil {
            return nil
        }

        var queue = [Node?]()
        queue.append(node)
        visited[node!.val] = Node(node!.val)

        while !queue.isEmpty {
            let temp = queue.removeFirst()
            for n in temp!.neighbors {
                if visited[n!.val] == nil {
                    visited[n!.val] = Node(n!.val)
                    queue.append(n)
                }
                visited[temp!.val]!!.neighbors.append(visited[n!.val]!)
            }
        }

        return visited[node!.val]!
    }
}
```
## 394. Decode String
### Use Stack
- time complexity: O(S), S为解码后的字符串长度
- space complexity: O(S)
```swift
class Solution {
    func decodeString(_ s: String) -> String {
        if s.isEmpty {
            return ""
        }
        
        var stack = [Character]()
        
        for c in s {
            if c != "]" {
                stack.append(c)
            } else {
                var tempStack = [Character]()
                while stack.last! != "[" {
                    tempStack.append(stack.removeLast())
                }
                stack.removeLast()  // remove "["
                tempStack.reverse()
                let tempStr = String(tempStack)
                
                // find k
                var numStack = [Character]()
                while !stack.isEmpty && stack.last!.isNumber {
                    numStack.append(stack.removeLast())
                }
                numStack.reverse()
                
                // duplicate tempStr k times
                let k = Int(String(numStack))!  
                var duplicateStr = ""
                for _ in 0 ..< k {
                    duplicateStr += tempStr
                }
                
                // append characters to stack
                for i in duplicateStr {
                    stack.append(i)
                }
                
            }
        }
        return String(stack)
    }
}
```
### Recursion
- time complexity: O(S), S为解码后的字符串长度
- space complexity: O(s), s为原字符串的长度
```swift
extension StringProtocol {
    subscript(offset: Int) -> Character {
        return self[index(startIndex, offsetBy: offset)]
    }
}

class Solution {
    
    func decodeString(_ s: String) -> String {
        var str = s
        var pointer = 0
        return getString(&str, &pointer)
    }
    
    func getString(_ str: inout String, _ pointer: inout Int) -> String {
        if pointer == str.count || str[pointer] == "]" {
            return ""
        }
        
        var res = ""
        var rep = 1
        let char = str[pointer]
        
        if char.isNumber {
            rep = getDigits(&str, &pointer)
            pointer += 1    // skip "["
            var string = getString(&str, &pointer)
            pointer += 1    // skip "]"
            while rep > 0 {
                res += string
                rep -= 1
            }
        } else if char.isLetter {
            res += String(char)
            pointer += 1
        }
        
        return res + getString(&str, &pointer)
    }
    
    func getDigits(_ str: inout String, _ pointer: inout Int) -> Int {
        var num = 0
        while pointer < str.count && str[pointer].isNumber {
            num = num * 10 + Int(String(str[pointer]))!
            pointer += 1
        }
        return num
    }
}
```
## 150. Evaluate Reverse Polish Notation
### Reducing list in place
- time complexity: O(n\*2)
- space complexity: O(1) 
```swift
class Solution {
    func evalRPN(_ tokens: [String]) -> Int {
        var arr = tokens
        var pointer = 0, length = arr.count
        let symbols: Set = ["+", "-", "*", "/"]
        
        while length > 1 {
            // jump to first operation symbol
            while !symbols.contains(arr[pointer]) {
                pointer += 1
            }
            
            let operation = arr[pointer]
            let num1 = Int(arr[pointer - 2])!
            let num2 = Int(arr[pointer - 1])!
            
            if operation == "+" {
                arr[pointer] = String(num1 + num2)
            } else if operation == "-" {
                arr[pointer] = String(num1 - num2)
            } else if operation == "*" {
                arr[pointer] = String(num1 * num2)
            } else if operation == "/" {
                arr[pointer] = String(num1 / num2)
            }
            
            // Delete previous two numbers
            for i in pointer-2 ..< length - 2 {
                arr[i] = arr[i + 2]
            }
            pointer -= 1
            length -= 2
        } 
        
        return Int(arr[0])!
    }
    
}
```
### Use stack
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func evalRPN(_ tokens: [String]) -> Int {
        var stack = [Int]()
        var operations: Set = ["*", "+", "-", "/"]
        for str in tokens {
            if !operations.contains(str) {
                stack.append(Int(str)!)
            } else {
                let num2 = stack.removeLast()
                let num1 = stack.removeLast()
                let result = applyOperation(str, num1, num2)
                stack.append(result)
            }
        }
        return stack.removeLast()
    }
    
    func applyOperation(_ operation: String, _ num1: Int, _ num2: Int) -> Int {
        var result = 0
        if operation == "+" {
            result = num1 + num2
        } else if operation == "-" {
            result = num1 - num2
        } else if operation == "*" {
            result = num1 * num2
        } else if operation == "/" {
            result = num1 / num2
        }
        return result
    }
}
```
## 155. Min Stack
- time complexity: O(1) for all operations
- space complexity: O(n)
```swift
class MinStack {

    /** initialize your data structure here. */
    var arr: [Int]
    var minValues: [Int]

    init() {
        arr = [Int]()
        minValues = [Int.max]
    }
    
    func push(_ x: Int) {
        arr.append(x)
        minValues.append(min(x, minValues.last!))
    }
    
    func pop() {
        arr.popLast()
        minValues.popLast()
    }
    
    func top() -> Int {
        arr.last!
    }
    
    func getMin() -> Int {
        return minValues.last!
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * let obj = MinStack()
 * obj.push(x)
 * obj.pop()
 * let ret_3: Int = obj.top()
 * let ret_4: Int = obj.getMin()
 */
```
## 143. Reorder List
将链表一分为二，然后将后半部分的链表反转与第一部分间序连接
- time complexity: O(n)
- space complexity: O(1)
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
    func reorderList(_ head: ListNode?) {
        if head == nil {
            return
        }
        
        let mid = getMid(head)
        let reversedHead = reverse(mid)
        
        var p1 = head, p2 = reversedHead
        while p2?.next != nil {
            var next = p1?.next
            p1?.next = p2
            p1 = next
            
            next = p2?.next
            p2?.next = p1
            p2 = next
        }
        
    }
    
    func reverse(_ head: ListNode?) -> ListNode? {
        var cur = head
        var prev: ListNode? = nil
        while cur != nil {
            let next = cur?.next
            cur?.next = prev
            prev = cur
            cur = next
        }
        return prev
    }
    
    func getMid(_ head: ListNode?) -> ListNode? {
        var slow = head, fast = head
        while fast != nil && fast?.next != nil {
            slow = slow?.next
            fast = fast?.next?.next
        }
        return slow
    }
}
```
## 148. Sort List
### Recursion
- time complexity: O(nlogn), each merge takes O(n) times, there are O(logn) times of merge, so totally O(nlogn)
- space complexity: O(logn)
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
    func sortList(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }
        let mid = getMid(head)
        let left = sortList(head)
        let right = sortList(mid)
        return merge(left, right)
    }
    
    func merge(_ node1: ListNode?, _ node2: ListNode?) -> ListNode? {
        var tempHead: ListNode? = ListNode(0), p1 = node1, p2 = node2
        var prev = tempHead
        while p1 != nil && p2 != nil {
            if p1!.val < p2!.val {
                prev?.next = p1
                prev = p1
                p1 = p1?.next
            } else {
                prev?.next = p2
                prev = p2
                p2 = p2?.next
            }
        }
        prev?.next = p1 ?? p2
        return tempHead?.next
    }
    
    func getMid(_ head: ListNode?) -> ListNode? {
        var slow = head, fast = head?.next
        while fast != nil && fast?.next != nil {
            slow = slow?.next
            fast = fast?.next?.next
        }
        let mid = slow?.next
        slow?.next = nil    // cut first half
        return mid  // return head of second half
    }
}
```
### Iteration
- time complexity: O(nlogn)
- space complexity: O(1)
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
    var tail: ListNode? = ListNode()
    var nextSubList: ListNode? = ListNode()
    
    func sortList(_ head: ListNode?) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }
        let n = getCount(head)
        var dummyHead: ListNode? = ListNode()
        dummyHead?.next = head
        var size = 1
        
        while size < n {
            tail = dummyHead
            var start = dummyHead?.next
            while start != nil {
                if start?.next == nil {
                    tail?.next = start
                    break
                }
                let mid = split(start, size)
                merge(start, mid)
                start = nextSubList
            }
            size *= 2
        }
        
        return dummyHead?.next
    }
    
    func merge(_ node1: ListNode?, _ node2: ListNode?) {        
        var tempHead: ListNode? = ListNode(), p1 = node1, p2 = node2
        var prev = tempHead
        while p1 != nil && p2 != nil {
            if p1!.val < p2!.val {
                prev?.next = p1
                p1 = p1?.next
            } else {
                prev?.next = p2
                p2 = p2?.next
            }
            prev = prev?.next
            
        }
        prev?.next = p1 ?? p2
        
        // find tail of merged list
        while prev?.next != nil {
            prev = prev?.next
        }
                
        tail?.next = tempHead?.next // link old tail to the head of merged list
        
        tail = prev // update tail to the new tail of merged list
    }
    
    func split(_ start: ListNode?, _ size: Int) -> ListNode? {
        var slow = start, fast = start?.next
        var index = 1
        while (slow?.next != nil || fast?.next != nil) && index < size {
            slow = slow?.next
            fast = fast?.next?.next
            index += 1
        }
        
        let mid = slow?.next    // track start of second linked list
        slow?.next = nil    // cut first linked list
        nextSubList = fast?.next    // track start of next sublist
        fast?.next = nil    // cut second linked list
        
        return mid  // return the start of second linked list
    }
    
    func getCount(_ head: ListNode?) -> Int {
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
## 86. Partition List
将链表分成比x小的和比x大的两个链表，再连接起来。*当头节点不确定时，使用dummyHead*
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
    func partition(_ head: ListNode?, _ x: Int) -> ListNode? {
        var before: ListNode? = ListNode(0), after: ListNode? = ListNode(0)
        var p1 = before, p2 = after
        var cur = head
        while cur != nil {
            let next = cur?.next
            if cur!.val < x {
                p1?.next = cur
                cur?.next = nil
                p1 = p1?.next
            } else {
                p2?.next = cur
                cur?.next = nil
                p2 = p2?.next
            }
            cur = next
        }
        // connect before and after lists
        p1?.next = after?.next

        return before?.next

    }
}
```
## 92. Reverse Linked List II
### Iteration
需要记录四个节点的位置，prev，cur，con和tail，*连接时需要考虑到链表的head被完全反转的corner case如([3, 5], 1, 2)这样的*。
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
    func reverseBetween(_ head: ListNode?, _ m: Int, _ n: Int) -> ListNode? {
        if head == nil || head?.next == nil {
            return head
        }
        
        // find start of reverse list
        var mIndex = m
        var prev: ListNode? = nil, cur = head
        while mIndex > 1 {
            prev = cur
            cur = cur?.next
            mIndex -= 1
        }
        let con = prev, tail = cur

        // reverse list
        var length = n - m + 1
        while length > 0 {
            let temp = cur?.next
            cur?.next = prev
            prev = cur
            cur = temp
            length -= 1
        }

        // connect reversed list and other parts
        tail?.next = cur
        if con == nil {   
            return prev
        } else {  
            con?.next = prev
            return head
        }   
    }
}
```
### Recursion + Backtracking
回溯是这个方法最难理解的点，关键在于newRight这个变量为局部变量，只存在于每一次的递归调用中，因此回溯时才能模拟右指针向左移动
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
    var left: ListNode?
    var isStop = false

    func reverseBetween(_ head: ListNode?, _ m: Int, _ n: Int) -> ListNode? {
        left = head
        recurse(head, m, n)
        return head
    }

    func recurse(_ right: ListNode?, _ m: Int, _ n: Int) {
        // stop when right pointer reaches nth node
        if n == 1 {
            return
        }
        
        var newRight = right?.next

        // stop when left pointer reaches mth node
        if m > 1 {
            left = left?.next
        }

        recurse(newRight, m - 1, n - 1)

        // check if backtracking go beyond left pointer
        if left === newRight || newRight?.next === left {
            isStop = true
        }

        // swap left and right pointer value
        if !isStop {
            let tempVal = newRight!.val
            newRight!.val = left!.val
            left!.val = tempVal
            // move left pointer forward, right pointer will move back with backtracking
            left = left?.next
        }
    }
}
```
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
- time complexity: O(n)
- space complexity: O(n)
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
        if root == nil {
            return TreeNode(val)
        }

        if root!.val > val {
            root?.left = insertIntoBST(root?.left, val)
        } else {
            root?.right = insertIntoBST(root?.right, val)
        }

        return root
    }
}
```
### Iteration
- time complexity: O(n)
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
        if root == nil {
            return TreeNode(val)
        }

        var cur = root
        while cur != nil {
            if cur!.val > val {
                if let left = cur?.left {
                    cur = left
                } else {
                    cur?.left = TreeNode(val)
                    break
                }
            } else {
                if let right = cur?.right {
                    cur = right
                } else {
                    cur?.right = TreeNode(val)
                    break
                }
            }
        }
        
        return root
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
        return isValidBST(root, Int.min, Int.max)
    }

    func isValidBST(_ root: TreeNode?, _ low: Int, _ high: Int) -> Bool {
        guard let root = root else {
            return true
        }

        if root.val <= low || root.val >= high {
            return false
        }

        return isValidBST(root.left, low, root.val) && isValidBST(root.right, root.val, high)
    }
}
```
### Inorder iteration
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
        var root = root
        var stack = [TreeNode?]()
        var inorderPrev = Int.min

        while !stack.isEmpty || root != nil {
            while root != nil {
                stack.append(root)
                root = root?.left
            }

            let node = stack.removeLast()
            if node!.val <= inorderPrev {
                return false
            }
            inorderPrev = node!.val
            root = node?.right
        }

        return true
    }
}
```
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
- time complexity: O(n^2), height会被重复调用因此耗时较高
- space complexity: O(n) 
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
    func isBalanced(_ root: TreeNode?) -> Bool {
        if root == nil {
            return true
        }

        if abs(height(root?.left) - height(root?.right)) > 1 {
            return false
        }

        return isBalanced(root?.left) && isBalanced(root?.right)
    }

    func height(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }

        return max(height(root?.left), height(root?.right)) + 1
    }
}
```
### bottom-up recursion
- time complexity: O(n)
- space complexity: O(n) 
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
    func isBalanced(_ root: TreeNode?) -> Bool {
        return height(root) != -1
    }
    
    func height(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }
        
        let left = height(root?.left)
        let right = height(root?.right)
        
        if (left == -1 || right == -1 || abs(left - right) > 1) {
            return -1
        }
        
        return max(left, right) + 1
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
### Hash Table
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        var dict = [Int: Int]()
        for n in nums {
            if dict[n] == nil {
                dict[n] = 0
            }
            dict[n]! += 1
        }
        for (val, freq) in dict {
            if freq == 1 {
                return val
            }
        }
        return 0
    }
}
```
### Math
2(a+b+c) - (a+a+b+b+c) = c
- time complexity: O(n)
- space complexity: O(n)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        var set = Set<Int>()
        var sumOfSet = 0, sumOfNums = 0
        for n in nums {
            if !set.contains(n) {
                set.insert(n)
                sumOfSet += n
            }
            sumOfNums += n
        }
        return 2 * sumOfSet - sumOfNums
    }
}
```
### XOR approach
- time complexity: O(n)
- space complexity: O(1)
```swift
class Solution {
    func singleNumber(_ nums: [Int]) -> Int {
        var res = 0
        for n in nums {
            res ^= n
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
- time complexity: O(m + n)
- space complexity: O(m) 
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
- time complexity: O(n + m)
- space complexity: O(1) 
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
### Recursion
- time complexity: O(n)
- space complexity: O(logn)
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
        if root == nil {
            return nil
        }
        connect(root?.left, root?.right)
        return root
    }

    func connect(_ node1: Node?, _ node2: Node?) {
        if node1 == nil || node2 == nil {
            return
        }

        node1?.next = node2
        connect(node1?.left, node1?.right)
        connect(node2?.left, node2?.right)
        connect(node1?.right, node2?.left)
    }
}
```
### Level-Order Traversal
- time complexity: O(n)
- space complexity: O(n), depending on the level that have max number of nodes
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
- time complexity: O(n)
- space complexity: O(n)
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
    var results = [[Int]]()
    
    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        if root == nil {
            return []
        }
        helper(root, 0)
        return results
    }
    
    func helper(_ node: TreeNode?, _ level: Int) {
        if level == results.count {
            results.append([])
        }
        results[level].append(node!.val)
        if let _ = node?.left {
            helper(node?.left, level + 1)
        }
        if let _ = node?.right {
            helper(node?.right, level + 1)
        }   
    }
}
```
### iteration
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
    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        if root == nil {
            return []
        }
        var results = [[Int]]()
        var queue = [TreeNode?]()
        queue.append(root)
        while !queue.isEmpty {
            let size = queue.count
            var levelValues = [Int]()
            for _ in 0..<size {
                let node = queue.removeFirst()
                levelValues.append(node!.val)
                if let left = node?.left {
                    queue.append(left)
                }
                if let right = node?.right {
                    queue.append(right)
                }
            }
            results.append(levelValues)
        }
        return results
    }
}
```
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
        guard let root = root else {
            return []
        }
        var res = [Int]()
        res += inorderTraversal(root.left)
        res.append(root.val)
        res += inorderTraversal(root.right)
        return res
    }
}
```
### Stack and Iteration
- time complexity: O(n)
- space complexity: O(n)
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
        var res = [Int]()
        var stack = [TreeNode?]()
        var cur = root
        while cur != nil || !stack.isEmpty {    
            while cur != nil {
                stack.append(cur)
                cur = cur?.left
            }
            cur = stack.removeLast()
            res.append(cur!.val)
            cur = cur?.right
        }
        return res
    }
}
```
### Morris Traversal
- time complexity: O(n)
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
    func inorderTraversal(_ root: TreeNode?) -> [Int] {
        var res = [Int]()
        var cur = root
        while cur != nil {
            if cur?.left != nil {
                var pre = cur?.left
                while pre?.right != nil {
                    pre = pre?.right
                }
                pre?.right = cur
                let temp = cur
                cur = cur?.left
                temp?.left = nil
            } else {
                res.append(cur!.val)
                cur = cur?.right
            }
        }
        return res
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
- time complexity: O(n)
- space complexity: O(n), 字典和递归的stack都会占用n的复杂度
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
    // holds old node as key, new node as value
    var visitedTable = [Node?: Node?]()
    
    func copyRandomList(_ head: Node?) -> Node? {
        if head == nil { return nil }
        
        // if we have processed the current node, then return the cloned version
        if visitedTable[head] != nil {
            return visitedTable[head]!
        }
        
        var node: Node? = Node(head!.val)  // copy the node
        visitedTable[head] = node
        
        node?.next = copyRandomList(head?.next)
        node?.random = copyRandomList(head?.random)
        
        return node
    }
}
```
### Iteration with O(n) space
- time complexity: O(n)
- space complexity: O(n)
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
    // holds old node as key, new node as value
    var visitedTable = [Node?: Node?]()
    
    func copyRandomList(_ head: Node?) -> Node? {
        if head == nil { return nil }
        
        var oldNode = head
        var newNode: Node? = Node(oldNode!.val)
        visitedTable[oldNode] = newNode
        
        while oldNode != nil {
            newNode?.next = getClonedNode(oldNode?.next)
            newNode?.random = getClonedNode(oldNode?.random)
            
            oldNode = oldNode?.next
            newNode = newNode?.next
        }
        
        return visitedTable[head]!
        
    }
    
    func getClonedNode(_ node: Node?) -> Node? {
        if node == nil { return nil }
        // if we have processed the current node, then return the cloned version
        if visitedTable[node] != nil {
            return visitedTable[node]!
        }
        
        var newNode: Node? = Node(node!.val)  // copy the node
        visitedTable[node] = newNode
        
        return newNode
    }
}
```
### Iteration with O(1) space
- time complexity: O(n)
- space complexity: O(1)
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
    func copyRandomList(_ head: Node?) -> Node? {
        if head == nil { return nil }
        
        // insert cloned node just next to the original node
        var oldNode = head
        while oldNode != nil {
            let copyNode: Node? = Node(oldNode!.val)
            let tempNext = oldNode?.next
            copyNode?.next = tempNext
            oldNode?.next = copyNode
            oldNode = tempNext
        }
        
        // copy random pointers
        oldNode = head
        while oldNode != nil {
            let clonedNode: Node? = oldNode?.next
            clonedNode?.random = oldNode?.random?.next
            oldNode = oldNode?.next?.next
        }
        
        // unweave and get back the origin list and its copy
        oldNode = head
        var copyHead = head?.next
        while oldNode != nil {
            let clonedNode: Node? = oldNode?.next
            oldNode?.next = clonedNode?.next
            clonedNode?.next = oldNode?.next?.next
            oldNode = oldNode?.next
        }
        
        return copyHead
        
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
- time complexity: O(max(m, n))
- space complexity: O(max(m, n)), 新链表长度为max(m, n)
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
    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        var carry = 0
        var p1 = l1, p2 = l2, dummyHead: ListNode? = ListNode(-1), prev = dummyHead
        var sum = 0
        while p1 != nil || p2 != nil {
            let val1 = p1 == nil ? 0 : p1!.val
            let val2 = p2 == nil ? 0 : p2!.val
            let sum = val1 + val2 + carry
            p1 = p1?.next
            p2 = p2?.next
            // append new node at tail
            carry = sum / 10
            let node = ListNode(sum % 10)
            prev?.next = node
            prev = node
        }

        if carry == 1 {
            prev?.next = ListNode(carry)
        }

        return dummyHead?.next
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
### HashTable
- time complexity: O(n)
- space complexity: O(n)
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
extension ListNode: Equatable {
    public static func ==(lhs: ListNode, rhs: ListNode) -> Bool {
        return lhs === rhs
    }
}
extension ListNode: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

class Solution {
    func hasCycle(_ head: ListNode?) -> Bool {
        if head == nil {
            return false
        }

        var set = Set<ListNode?>()
        var cur = head
        while cur != nil {
            if set.contains(cur) {
                return true
            }
            set.insert(cur)
            cur = cur?.next
        }

        return false
    }
}
```
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
        var slow = head, fast = head?.next
        while fast != nil && fast?.next != nil {
            if slow === fast {
                return true
            }
            slow = slow?.next
            fast = fast?.next?.next
        }
        return false
    }
}
```

## 142. Linked List Cycle II
### HashTable
- time complexity: O(n)
- space complexity: O(n)
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
extension ListNode: Equatable {
    public static func == (lhs: ListNode, rhs: ListNode) -> Bool {
        return lhs === rhs
    }
}
extension ListNode: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

class Solution {
    func detectCycle(_ head: ListNode?) -> ListNode? {
        if head == nil {
            return head
        }
        var cur = head
        var hashSet: Set<ListNode?> = []
        while cur != nil {
            if hashSet.contains(cur) {
                return cur
            }
            hashSet.insert(cur)
            cur = cur?.next
        }
        return nil
    }
}
```
### Two Pointers
如果有环的话先找到交点，在用两个指针指向head和交点，一次只移动一步，指针相遇的地方即为环的起始点
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
    func detectCycle(_ head: ListNode?) -> ListNode? {
        if head == nil {
            return nil
        }
        // find the meeting point of slow and fast pointers
        var intersect = getIntersect(head)
        if intersect == nil {
            return nil
        }

        // there must be a cycle
        var start = head
        while start !== intersect {
            start = start?.next
            intersect = intersect?.next
        }
        return start
    }

    func getIntersect(_ head: ListNode?) -> ListNode? {
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
- time complexity: O(L)
- space complexity: O(1)
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
        if head == nil {
            return nil
        }
        
        var dummyHead: ListNode? = ListNode()
        var cur = dummyHead
        dummyHead?.next = head
        var length = getLength(head)
        var i = 0
        while i < length - n {
            cur = cur?.next
            i += 1
        }

        cur?.next = cur?.next?.next

        return dummyHead?.next
    }

    func getLength(_ head: ListNode?) -> Int {
        var cur = head
        var count = 0
        while cur != nil {
            count += 1
            cur = cur?.next
        }
        return count
    }
}
```
### One Pass Algorithm
- time complexity: O(L)
- space complexity: O(1)
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
        if head == nil {
            return nil
        }

        var dummyHead: ListNode? = ListNode()
        dummyHead?.next = head
        var first = dummyHead, second = dummyHead

        // move second n+1 steps
        var i = 0
        while i < n + 1 {
            first = first?.next
            i += 1
        }

        while first != nil {
            second = second?.next
            first = first?.next
        }

        second?.next = second?.next?.next

        return dummyHead?.next
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
        if head?.next == nil || head == nil {
            return head
        }

        let node = reverseList(head?.next)
        head?.next?.next = head
        head?.next = nil

        return node
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
        if head == nil {
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
### Recursion
- time complexity: O(n)
- space complexity: O(n)
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
    var frontPointer: ListNode? = nil

    func isPalindrome(_ head: ListNode?) -> Bool {
        frontPointer = head
        return recurseCheck(head)
    }

    func recurseCheck(_ current: ListNode?) -> Bool {
        if current != nil {
            if !recurseCheck(current?.next) {
                return false
            }
            if frontPointer!.val != current!.val {
                return false
            }
            frontPointer = frontPointer?.next
        }
        return true
    }
}
```
### Copy Linked List into array and use two pointers approach
- time complexity: O(n)
- space complexity: O(n)
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
    func isPalindrome(_ head: ListNode?) -> Bool {
        if head == nil || head?.next == nil {
            return true
        }
        // copy all values to array
        var values = [Int]()
        var cur = head
        while cur != nil {
            values.append(cur!.val)
            cur = cur?.next
        }
        
        var left = 0, right = values.count - 1 
        while left < right {
            if values[left] != values[right] {
                return false
            }
            left += 1
            right -= 1
        }
        return true

    }
}
```
### Reverse Second-half in place
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
    func isPalindrome(_ head: ListNode?) -> Bool {
        if head == nil || head?.next == nil {
            return true
        }

        var fistPartEnd = findMid(head)
        var secondPartHead = reverse(fistPartEnd?.next)

        var result = true
        var p1 = head, p2 = secondPartHead
        while result && p2 != nil {
            if p1!.val != p2!.val {
                result = false
            }
            p1 = p1?.next
            p2 = p2?.next
        }

        // restore original list
        fistPartEnd?.next = reverse(secondPartHead)

        return result
    }

    func reverse(_ head: ListNode?) -> ListNode? {
        var prev: ListNode? = nil, cur = head
        while cur != nil {
            let temp = cur?.next
            cur?.next = prev
            prev = cur
            cur = temp
        }
        return prev
    }

    func findMid(_ head: ListNode?) -> ListNode? {
        var slow = head, fast = head?.next
        while fast != nil && fast?.next != nil {
            slow = slow?.next
            fast = fast?.next?.next
        }
        return slow
    }
}
```
## 21. Merge Two Sorted Lists
### Recursion
- time complexity: O(m+n)
- space complexity: O(m+n)
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
        if l1 == nil {
            return l2
        } 
        if l2 == nil {
            return l1
        }
        var p1 = l1, p2 = l2 
        if p1!.val <= p2!.val {
            p1?.next = mergeTwoLists(p1?.next, p2)
            return p1
        } else {
            p2?.next = mergeTwoLists(p1, p2?.next)
            return p2
        }
    }
}
```
### Iteration
使用伪head节点来方便返回head
- time complexity: O(m+n)
- space complexity: O(1)
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
        var dummyHead: ListNode? = ListNode()
        var p1 = l1, p2 = l2, prev = dummyHead
        while p1 != nil && p2 != nil {
            if p1!.val <= p2!.val {
                prev?.next = p1
                prev = p1
                p1 = p1?.next
            } else {
                prev?.next = p2
                prev = p2
                p2 = p2?.next
            }
        }
        prev?.next = p1 ?? p2
        return dummyHead?.next
    }
}
```
