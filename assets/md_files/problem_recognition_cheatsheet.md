# Problem Recognition & Solutions Cheat Sheet

Below is a **cleaned + expanded** version where **each problem includes its solution immediately after**, so you can revise ultra-fast before your interview.

---

# 0. Data Structures & APIs Reference

## Dictionary/HashMap (`dict`)
**Operations:**
- `d[key] = value` - Insert/Update: O(1) avg, O(n) worst
- `d[key]` - Get: O(1) avg, O(n) worst
- `d.get(key, default)` - Get with default: O(1) avg, O(n) worst
- `key in d` - Check existence: O(1) avg, O(n) worst
- `d.pop(key)` - Remove: O(1) avg, O(n) worst
- `len(d)` - Size: O(1)
- `d.keys()`, `d.values()`, `d.items()` - Iterate: O(n)

**Space:** O(n) where n = number of key-value pairs

**Examples:**
```python
d = {}
d['a'] = 1          # O(1)
d['b'] = 2          # O(1)
x = d['a']          # O(1)
if 'c' in d: ...    # O(1)
d.pop('a')          # O(1)
```

---

## List/Array (`list`)
**Operations:**
- `arr[i]` - Access by index: O(1)
- `arr.append(x)` - Add to end: O(1) amortized
- `arr.pop()` - Remove from end: O(1)
- `arr.pop(i)` - Remove at index i: O(n)
- `arr.insert(i, x)` - Insert at index i: O(n)
- `arr.remove(x)` - Remove first occurrence: O(n)
- `arr.index(x)` - Find index: O(n)
- `arr.sort()` - Sort in-place: O(n log n)
- `sorted(arr)` - Return sorted copy: O(n log n)
- `len(arr)` - Size: O(1)
- `arr[i:j]` - Slice: O(j-i)

**Space:** O(n) where n = length of list

**Examples:**
```python
arr = [1, 2, 3]
arr.append(4)       # O(1)
x = arr[0]          # O(1)
arr.pop()           # O(1)
arr.insert(0, 0)    # O(n)
arr.sort()          # O(n log n)
```

---

## Set (`set`)
**Operations:**
- `s.add(x)` - Add element: O(1) avg, O(n) worst
- `s.remove(x)` - Remove element: O(1) avg, O(n) worst
- `x in s` - Check membership: O(1) avg, O(n) worst
- `s.pop()` - Remove arbitrary element: O(1) avg
- `len(s)` - Size: O(1)
- `s1 | s2` - Union: O(len(s1) + len(s2))
- `s1 & s2` - Intersection: O(min(len(s1), len(s2)))
- `s1 - s2` - Difference: O(len(s1))

**Space:** O(n) where n = number of elements

**Examples:**
```python
s = set()
s.add(1)            # O(1)
s.add(2)            # O(1)
if 1 in s: ...      # O(1)
s.remove(1)         # O(1)
```

---

## Stack (`list` used as stack)
**Operations:**
- `stack.append(x)` - Push: O(1)
- `stack.pop()` - Pop: O(1)
- `stack[-1]` - Peek top: O(1)
- `len(stack)` - Size: O(1)

**Space:** O(n) where n = number of elements

**Examples:**
```python
stack = []
stack.append(1)     # O(1)
stack.append(2)     # O(1)
top = stack[-1]     # O(1)
x = stack.pop()     # O(1)
```

---

## Queue (`collections.deque`)
**Operations:**
- `q.append(x)` - Enqueue (add to right): O(1)
- `q.appendleft(x)` - Add to left: O(1)
- `q.popleft()` - Dequeue (remove from left): O(1)
- `q.pop()` - Remove from right: O(1)
- `q[0]` - Peek front: O(1)
- `len(q)` - Size: O(1)

**Space:** O(n) where n = number of elements

**Examples:**
```python
from collections import deque
q = deque()
q.append(1)         # O(1)
q.append(2)         # O(1)
x = q.popleft()     # O(1)
```

---

## Heap/Priority Queue (`heapq`)
**Operations:**
- `heapq.heappush(heap, item)` - Insert: O(log n)
- `heapq.heappop(heap)` - Remove min: O(log n)
- `heap[0]` - Peek min: O(1)
- `heapq.heapify(arr)` - Build heap: O(n)
- `heapq.nlargest(k, arr)` - K largest: O(n log k)
- `heapq.nsmallest(k, arr)` - K smallest: O(n log k)

**Space:** O(n) where n = number of elements

**Examples:**
```python
import heapq
heap = []
heapq.heappush(heap, 3)  # O(log n)
heapq.heappush(heap, 1)  # O(log n)
min_val = heap[0]        # O(1)
x = heapq.heappop(heap)  # O(log n)
```

---

## Binary Search Tree (`bisect` on sorted list)
**Operations:**
- `bisect.bisect_left(arr, x)` - Find leftmost position: O(log n)
- `bisect.bisect_right(arr, x)` - Find rightmost position: O(log n)
- `bisect.insort_left(arr, x)` - Insert maintaining order: O(n) (due to shift)
- `bisect.insort_right(arr, x)` - Insert maintaining order: O(n)

**Space:** O(n) where n = length of sorted list

**Examples:**
```python
import bisect
arr = [1, 3, 5, 7]
pos = bisect.bisect_left(arr, 4)  # O(log n) → returns 2
bisect.insort_left(arr, 4)        # O(n) → [1, 3, 4, 5, 7]
```

---

## Graph (Adjacency List)
**Representation:**
```python
graph = {
    'A': [('B', 1), ('C', 2)],
    'B': [('D', 3)],
    'C': [('D', 1)]
}
```

**Operations:**
- Access neighbors: O(1) to O(V) depending on degree
- Add edge: O(1)
- Check edge existence: O(degree)

**Space:** O(V + E) where V = vertices, E = edges

---

## Trie (`dict`-based)
**Operations:**
- Insert word: O(m) where m = word length
- Search word: O(m)
- Search prefix: O(m)
- Delete word: O(m)

**Space:** O(ALPHABET_SIZE * N * M) where N = number of words, M = average length

**Example:**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

def insert(root, word):  # O(m)
    node = root
    for ch in word:
        if ch not in node.children:
            node.children[ch] = TrieNode()
        node = node.children[ch]
    node.is_end = True
```

---

# 1. Pattern Recognition Rules (Cheat Sheet)
- **Sliding Window** → contiguous subarray problems (longest/shortest window, positive nums).
- **Two Pointers** → sorted arrays, convergence, subsequence-like problems.
- **Prefix Sum + HashMap** → subarray sums with **negative numbers**.
- **Binary Search on Answer** → minimize/maximize capacity, time, speed, etc.
- **Greedy** → local optimal choices provably lead to global optimum.
- **DP** → optimal substructure, overlapping subproblems.
- **Monotonic Stack** → next greater/smaller, histogram, temperatures.
- **Heap (Priority Queue)** → need smallest/largest item repeatedly.
- **Graph BFS/DFS** → grid problems, islands, shortest path (unweighted).
- **Tree DFS with state** → diameter, max path sum.

---

# 2. Problems & Solutions (One place)

---
## 1. Longest Substring Without Repeating Characters
**Pattern:** Sliding Window  
**DS:** `dict` (char → last index)  
**Intuition:** Maintain a sliding window with two pointers. When we see a duplicate character, move the left pointer past the last occurrence of that character. Track the maximum window size.  
**Time:** O(n) - each character visited at most twice  
**Space:** O(min(n, m)) - where m is the size of the character set

**Solution:**
```python
l = 0
seen = {}
best = 0
for r, ch in enumerate(s):
    if ch in seen and seen[ch] >= l:
        l = seen[ch] + 1
    seen[ch] = r
    best = max(best, r-l+1)
return best
```

---
## 2. Minimum Length Subarray with Sum ≥ S
**Pattern:** Sliding Window  
**DS:** none  
**Intuition:** Expand window by moving right pointer, adding elements. When sum ≥ S, try to shrink from left while maintaining the condition. Track minimum window size.  
**Time:** O(n) - each element visited at most twice  
**Space:** O(1)

**Solution:**
```python
l = 0
cur = 0
ans = float('inf')
for r, x in enumerate(nums):
    cur += x
    while cur >= S:
        ans = min(ans, r-l+1)
        cur -= nums[l]
        l += 1
return ans if ans != float('inf') else 0
```

---
## 3. Two-Sum in Sorted Array
**Pattern:** Two Pointers  
**DS:** none  
**Intuition:** Since array is sorted, start with pointers at both ends. If sum is too small, move left pointer right (increase sum). If too large, move right pointer left (decrease sum).  
**Time:** O(n)  
**Space:** O(1)

**Solution:**
```python
l, r = 0, len(nums)-1
while l < r:
    s = nums[l] + nums[r]
    if s == target: return True
    if s < target: l += 1
    else: r -= 1
return False
```

---
## 4. Count Subarrays with Sum == K
**Pattern:** Prefix Sum + HashMap  
**DS:** `dict`  
**Intuition:** Use prefix sums. For each position, if prefix_sum - K exists in map, it means there's a subarray ending here with sum K. Track frequency of each prefix sum.  
**Time:** O(n)  
**Space:** O(n) - for the hashmap

**Solution:**
```python
mp = {0:1}
p = 0
ans = 0
for x in nums:
    p += x
    ans += mp.get(p-K, 0)
    mp[p] = mp.get(p, 0) + 1
return ans
```

---
## 5. Weighted Interval Scheduling (Max Profit)
**Pattern:** DP + Binary Search  
**DS:** sorted list + `bisect`  
**Intuition:** Sort intervals by end time. For each interval, decide: include it (add profit + best profit from non-overlapping intervals) or exclude it. Use binary search to find the last non-overlapping interval efficiently.  
**Time:** O(n log n) - sorting + n binary searches  
**Space:** O(n) - for dp array and sorted ends

**Solution:**
```python
intervals.sort(key=lambda x: x[1])
ends = [e for _,e,_ in intervals]
dp = [0]*len(intervals)

for i,(s,e,p) in enumerate(intervals):
    j = bisect.bisect_right(ends, s) - 1
    include = p + (dp[j] if j>=0 else 0)
    exclude = dp[i-1] if i>0 else 0
    dp[i] = max(include, exclude)
return dp[-1]
```

---
## 6. House Robber
**Pattern:** DP  
**DS:** none  
**Intuition:** At each house, we can either rob it (add to profit from 2 houses ago) or skip it (keep profit from previous house). Choose the maximum. Only need to track last two states.  
**Time:** O(n)  
**Space:** O(1) - only storing prev2 and prev1

**Solution:**
```python
prev2 = prev1 = 0
for x in nums:
    cur = max(prev1, prev2 + x)
    prev2, prev1 = prev1, cur
return prev1
```

---
## 7. Largest Rectangle in Histogram
**Pattern:** Monotonic Stack  
**DS:** stack  
**Intuition:** Use a stack to track indices of bars in increasing height order. When we see a bar shorter than stack top, we know the rectangle height for stack top can't extend further. Calculate area using the popped bar's height and the width from previous smaller bar to current position.  
**Time:** O(n) - each bar pushed and popped once  
**Space:** O(n) - for the stack

**Solution:**
```python
stack = []
ans = 0
heights.append(0)
for i,h in enumerate(heights):
    while stack and heights[stack[-1]] > h:
        H = heights[stack.pop()]
        L = stack[-1] if stack else -1
        ans = max(ans, H * (i-L-1))
    stack.append(i)
return ans
```

---
## 8. Shortest Path in Graph (Non-negative Weights)
**Pattern:** Dijkstra  
**DS:** Min-heap  
**Intuition:** Start from source with distance 0. Use a min-heap to always process the node with smallest distance first. For each neighbor, if we found a shorter path, update distance and add to heap. This guarantees we find shortest paths in order.  
**Time:** O((V + E) log V) - each edge processed once, heap operations are log V  
**Space:** O(V) - for dist map and heap

**Solution:**
```python
pq = [(0, src)]
dist = {src:0}
while pq:
    d, node = heapq.heappop(pq)
    if d > dist[node]: continue
    for nei,w in graph[node]:
        nd = d + w
        if nd < dist.get(nei, float('inf')):
            dist[nei] = nd
            heapq.heappush(pq, (nd, nei))
return dist
```

---
## 9. Minimum Ship Capacity to Ship in D Days
**Pattern:** Binary Search on Answer + Greedy  
**DS:** none  
**Intuition:** The answer (capacity) is in a range [max(weights), sum(weights)]. Binary search on this range. For each capacity, greedily pack items and count days needed. If days ≤ D, capacity is valid (try smaller). Otherwise, need larger capacity.  
**Time:** O(n log(sum(weights) - max(weights))) - binary search * greedy check  
**Space:** O(1)

**Solution:**
```python
def can(cap):
    days = 1
    cur = 0
    for w in weights:
        if cur + w > cap:
            days += 1
            cur = 0
        cur += w
    return days <= D

l, r = max(weights), sum(weights)
while l < r:
    mid = (l+r)//2
    if can(mid): r = mid
    else: l = mid+1
return l
```

---
## 10. Is Subsequence
**Pattern:** Two Pointers  
**DS:** none  
**Intuition:** Use one pointer for each string. Move pointer in t forward always. Move pointer in s forward only when characters match. If s pointer reaches end, s is a subsequence of t.  
**Time:** O(n) where n = len(t)  
**Space:** O(1)

**Solution:**
```python
i = 0
for ch in t:
    if i < len(s) and s[i] == ch:
        i += 1
return i == len(s)
```

---
## 11. Maximum Subarray Sum (Kadane)
**Pattern:** DP/Greedy  
**DS:** none  
**Intuition:** At each position, decide: start a new subarray from here, or extend the previous subarray. Extend only if previous sum is positive (adds value). Track the maximum sum seen so far.  
**Time:** O(n)  
**Space:** O(1)

**Solution:**
```python
cur = best = nums[0]
for x in nums[1:]:
    cur = max(x, cur+x)
    best = max(best, cur)
return best
```

---
## 12. Merge Intervals
**Pattern:** Sort + Sweep  
**DS:** list  
**Intuition:** Sort intervals by start time. Iterate through sorted intervals. If current interval overlaps with last merged interval (start ≤ last end), merge by extending the end. Otherwise, add as new interval.  
**Time:** O(n log n) - dominated by sorting  
**Space:** O(n) - for result array (worst case no merges)

**Solution:**
```python
intervals.sort()
res = [intervals[0]]
for s,e in intervals[1:]:
    if s <= res[-1][1]:
        res[-1][1] = max(res[-1][1], e)
    else:
        res.append([s,e])
return res
```

---
## 13. Best Time to Buy/Sell Stock (1 Transaction)
**Pattern:** Greedy  
**DS:** none  
**Intuition:** Track the minimum price seen so far. At each day, calculate profit if we sell today (price - min_price). Update maximum profit. We only need to remember the minimum price and best profit.  
**Time:** O(n)  
**Space:** O(1)

**Solution:**
```python
minp = float('inf')
profit = 0
for x in prices:
    minp = min(minp, x)
    profit = max(profit, x-minp)
return profit
```

---
## 14. Longest Palindromic Substring
**Pattern:** Expand Centers  
**DS:** none  
**Intuition:** Every palindrome expands from a center. Check both odd-length (center at one char) and even-length (center between two chars) palindromes. For each position, expand outward while characters match. Track the longest found.  
**Time:** O(n²) - n positions, each expansion can be O(n)  
**Space:** O(1) - only storing start/end indices

**Solution:**
```python
def expand(l,r):
    while l>=0 and r<len(s) and s[l]==s[r]:
        l-=1; r+=1
    return r-l-1

start = end = 0
for i in range(len(s)):
    L = max(expand(i,i), expand(i,i+1))
    if L > end-start+1:
        start = i - (L-1)//2
        end = i + L//2
return s[start:end+1]
```

---
## 15. Number of Islands
**Pattern:** DFS/BFS  
**DS:** recursion or queue  
**Intuition:** Iterate through grid. When we find land ('1'), mark all connected land cells as visited using DFS/BFS. Each connected component is one island. Count how many times we find unvisited land.  
**Time:** O(m × n) - visit each cell once  
**Space:** O(m × n) - recursion stack/queue in worst case (all land)

**Solution:**
```python
def dfs(i, j):
    if i<0 or i>=m or j<0 or j>=n or grid[i][j]=='0':
        return
    grid[i][j] = '0'  # mark as visited
    dfs(i+1, j); dfs(i-1, j); dfs(i, j+1); dfs(i, j-1)

count = 0
for i in range(m):
    for j in range(n):
        if grid[i][j]=='1':
            dfs(i, j)
            count += 1
return count
```

---
## 16. Burst Balloons (Max Coins)
**Pattern:** Interval DP  
**DS:** 2D dp  
**Intuition:** Instead of thinking which balloon to burst first, think which balloon to burst LAST in a range. If balloon k is the last in range [l, r], coins = nums[l] × nums[k] × nums[r] + solve(l, k) + solve(k, r). Try all possible last balloons.  
**Time:** O(n³) - O(n²) subproblems, each takes O(n) to find best k  
**Space:** O(n²) - for dp table

**Solution:**
```python
nums = [1] + nums + [1]
n = len(nums)
dp = [[0]*n for _ in range(n)]
for length in range(2, n):
    for l in range(0, n-length):
        r = l+length
        for k in range(l+1, r):
            dp[l][r] = max(dp[l][r], nums[l]*nums[k]*nums[r] + dp[l][k] + dp[k][r])
return dp[0][n-1]
```

---
## 17. Cycle Detection (Directed)
**Pattern:** DFS with recursion stack  
**DS:** adjacency list + visited + onpath  
**Intuition:** Use DFS with two sets: `visited` (all nodes seen) and `onpath` (nodes in current recursion path). If we encounter a node that's in `onpath`, we found a back edge → cycle. Remove from `onpath` when backtracking.  
**Time:** O(V + E) - standard DFS  
**Space:** O(V) - for visited and onpath sets, recursion stack

**Solution:**
```python
def dfs(u):
    visited.add(u); onpath.add(u)
    for v in g[u]:
        if v not in visited:
            if dfs(v): return True
        elif v in onpath:
            return True
    onpath.remove(u)
    return False
```

---
## 18. Edit Distance
**Pattern:** 2D DP  
**DS:** matrix  
**Intuition:** `dp[i][j]` = min edits to convert `a[0:i]` to `b[0:j]`. If characters match, no edit needed (use diagonal). Otherwise, take minimum of: insert (left), delete (top), or replace (diagonal). Base case: converting empty string requires length of other string.  
**Time:** O(m × n) where m = len(a), n = len(b)  
**Space:** O(m × n) - can be optimized to O(min(m, n))

**Solution:**
```python
dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(m+1): dp[i][0] = i
for j in range(n+1): dp[0][j] = j
for i in range(1,m+1):
    for j in range(1,n+1):
        if a[i-1]==b[j-1]: 
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j],    # delete
                              dp[i][j-1],      # insert
                              dp[i-1][j-1])    # replace
return dp[m][n]
```

---

## 19. Meeting Rooms II
**Pattern:** Min-heap / Sweep-line  
**DS:** heap  
**Intuition:** Sort meetings by start time. Use a min-heap to track end times of ongoing meetings. For each meeting, if the earliest ending meeting has finished (heap top ≤ current start), reuse that room (pop). Otherwise, need a new room. Always push current meeting's end time.  
**Time:** O(n log n) - sorting + n heap operations  
**Space:** O(n) - for heap in worst case

**Solution:**
```python
import heapq
intervals.sort(key=lambda x: x[0])
rooms = []
for s, e in intervals:
    if rooms and rooms[0] <= s:
        heapq.heappop(rooms)
    heapq.heappush(rooms, e)
return len(rooms)
```

---

## 20. Longest Increasing Subsequence
**Pattern:** DP (O(n²)) or Patience Sorting (O(n log n))  
**DS:** list + `bisect`  
**Intuition (O(n log n)):** Maintain an array `tails` where `tails[i]` is the smallest tail of all increasing subsequences of length `i+1`. For each number, binary search to find where it should go. If it's larger than all tails, extend the sequence. Otherwise, replace the first tail that's ≥ current number (maintains smaller tail for future).  
**Time:** O(n log n) - n elements, each binary search is O(log n)  
**Space:** O(n) - for tails array

**Solution (O(n log n)):**
```python
from bisect import bisect_left
tails = []
for x in nums:
    i = bisect_left(tails, x)
    if i == len(tails):
        tails.append(x)
    else:
        tails[i] = x
return len(tails)
```

---

## 21. Binary Tree Maximum Path Sum
**Pattern:** Tree DP (postorder DFS)  
**DS:** recursion  
**Intuition:** For each node, calculate the maximum path sum that goes through it. A path through a node can either: (1) go down from node (single branch), or (2) go through node connecting left and right branches. Return the maximum single-branch path to parent (can't use both branches when returning). Track global maximum.  
**Time:** O(n) - visit each node once  
**Space:** O(h) - recursion stack, where h is tree height

**Solution:**
```python
def dfs(node):
    if not node: return 0
    left = max(0, dfs(node.left))    # ignore negative paths
    right = max(0, dfs(node.right))
    # path through this node connecting both branches
    self.ans = max(self.ans, node.val + left + right)
    # return max single branch path
    return node.val + max(left, right)

self.ans = float('-inf')
dfs(root)
return self.ans
```

---

## 22. Subarrays with Product < K
**Pattern:** Sliding Window (positive nums only)  
**DS:** none  
**Intuition:** Use sliding window with two pointers. Expand window by moving right pointer, multiply product. When product ≥ K, shrink from left by dividing. Count subarrays ending at right pointer: all subarrays from left to right are valid (right - left + 1).  
**Time:** O(n) - each element visited at most twice  
**Space:** O(1)

**Solution:**
```python
l = 0
prod = 1
ans = 0
for r in range(len(nums)):
    prod *= nums[r]
    while l <= r and prod >= k:
        prod //= nums[l]
        l += 1
    ans += r - l + 1
return ans
```

---

## 23. Jump Game I
**Pattern:** Greedy (track farthest reach)  
**DS:** none  
**Intuition:** Track the farthest index we can reach. Iterate through array. If current index > farthest reach, we can't get here → return False. Otherwise, update farthest reach = max(farthest, i + nums[i]). If farthest ≥ last index, return True.  
**Time:** O(n)  
**Space:** O(1)

**Solution:**
```python
farthest = 0
for i in range(len(nums)):
    if i > farthest:
        return False
    farthest = max(farthest, i + nums[i])
    if farthest >= len(nums) - 1:
        return True
return True
```

---

## 24. Jump Game II
**Pattern:** Greedy (window-based BFS-level approach)  
**DS:** none  
**Intuition:** Think of it as BFS levels. At each level, find the farthest we can reach. Start at level 0 (index 0). For current level, find the farthest reachable index (end of current level). That becomes the start of next level. Count how many levels needed to reach end.  
**Time:** O(n)  
**Space:** O(1)

**Solution:**
```python
jumps = 0
cur_end = 0
farthest = 0
for i in range(len(nums) - 1):
    farthest = max(farthest, i + nums[i])
    if i == cur_end:
        jumps += 1
        cur_end = farthest
        if cur_end >= len(nums) - 1:
            break
return jumps
```

---

## 25. Gas Station
**Pattern:** Greedy + running total  
**DS:** none  
**Intuition:** If total gas < total cost, impossible. Otherwise, a solution exists. Start from index 0, track running total (gas - cost). If running total becomes negative at index i, all stations from start to i are invalid starting points (they all lead to negative). Start from i+1 and reset running total.  
**Time:** O(n)  
**Space:** O(1)

**Solution:**
```python
total = 0
cur = 0
start = 0
for i in range(len(gas)):
    diff = gas[i] - cost[i]
    total += diff
    cur += diff
    if cur < 0:
        start = i + 1
        cur = 0
return start if total >= 0 else -1
```

---

## 26. Coin Change (Min Coins)
**Pattern:** DP (unbounded knapsack)  
**DS:** 1D dp array  
**Intuition:** `dp[i]` = minimum coins needed to make amount `i`. For each amount, try each coin. If coin value ≤ amount, `dp[i] = min(dp[i], 1 + dp[i - coin])`. Initialize dp[0] = 0, others = infinity.  
**Time:** O(amount × len(coins))  
**Space:** O(amount)

**Solution:**
```python
dp = [float('inf')] * (amount + 1)
dp[0] = 0
for i in range(1, amount + 1):
    for coin in coins:
        if coin <= i:
            dp[i] = min(dp[i], 1 + dp[i - coin])
return dp[amount] if dp[amount] != float('inf') else -1
```

---

## 27. Partition Equal Subset Sum
**Pattern:** Subset Sum DP  
**DS:** 1D dp or 2D prev/curr  
**Intuition:** If total sum is odd, impossible. Otherwise, check if we can form sum/2 using subset. `dp[i]` = True if we can form sum `i` using some subset. For each number, iterate backwards through dp array to avoid using same number twice.  
**Time:** O(n × sum)  
**Space:** O(sum) - can be optimized from O(n × sum)

**Solution:**
```python
total = sum(nums)
if total % 2: return False
target = total // 2
dp = [False] * (target + 1)
dp[0] = True
for num in nums:
    for i in range(target, num - 1, -1):
        dp[i] = dp[i] or dp[i - num]
return dp[target]
```

---

## 28. K-Consecutive Bit Flips
**Pattern:** Greedy + flip parity tracking  
**DS:** array + in-place markers  
**Intuition:** Greedily flip whenever we see a 0. Use a variable to track if current position has been flipped (even/odd number of times). When we flip a window starting at i, mark that position i+k will be affected (flip parity changes). Use XOR or a boolean array to track flip state.  
**Time:** O(n)  
**Space:** O(n) - for flip tracking, can be optimized to O(1) with sliding window

**Solution:**
```python
flipped = [False] * len(nums)
ans = 0
flip_count = 0
for i in range(len(nums)):
    if i >= k:
        flip_count ^= flipped[i - k]  # remove effect of old flip
    if flip_count == nums[i]:  # need to flip
        if i + k > len(nums):
            return -1
        flipped[i] = True
        flip_count ^= 1
        ans += 1
return ans
```