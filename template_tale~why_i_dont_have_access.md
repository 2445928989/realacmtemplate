# 基础

###### 二分

```cpp
int l = 左边界, r = 右边界;(闭区间)
int ans = 0;
while (l <= r) {
	int mid = (l + r) >> 1;
	if (check()) {
		l = mid + 1;
		ans = mid;
	}
	else {
		r = mid - 1;
	}
}
//do...
```

###### 三分

```cpp
int l = 左边界, r = 右边界;(闭区间)
int ans = 0;
while (l <= r) {
	int lmid = max(((l + r) >> 1) - 1, l);
	int rmid = min(((l + r) >> 1) + 1, r);
	if (f(lmid) < f(rmid))l = lmid + 1;
	else r = rmid - 1;
	ans = max({ ans, f(lmid), f(rmid) });
}
//寻找极值
```



# 数据结构

###### 并查集

```cpp
struct DSU
{
    vector<int> fa, sz;
    DSU(int n)
    {
        fa.resize(n + 1);
        sz.resize(n + 1);
        init(n);
    }
    void init(int n)
    {
        for (int i = 1; i <= n; i++)
        {
            fa[i] = i;
            sz[i] = 1;
        }
    }
    int find(int a)
    {
        if (fa[a] == a)
            return a;
        else
            return fa[a] = find(fa[a]);
    }
    int merge(int x, int y)
    { // 启发，让大小比较小的合并到大小比较大的上。
        int fx = find(x), fy = find(y);
        if (fx == fy)
            return sz[fx];
        if (sz[fx] < sz[fy])
            swap(fx, fy);
        fa[fy] = fx;
        sz[fx] += sz[fy];
        return sz[fx];
    }
};
```

###### 线性基

```cpp
struct Linear_basis
{
    vector<long long> p;
    int n;
    bool flag = 0; // 有无0
    void insert(long long x)
    {
        for (int i = n - 1; ~i; i--)
        {
            if (!(x >> i)) // 第i位是0
                continue;
            if (!p[i])
            {
                p[i] = x;
                if (!x)
                    flag = 1;
                return;
            }
            x ^= p[i];
        }
    }
    long long get_max(int x = 0)
    {
        int ret = x;
        for (int i = n - 1; ~i; i--)
        {
            ret = max(ret, ret ^ p[i]);
        }
        return ret;
    }
    long long get_min()
    {
        if (flag)
            return 0;
        for (int i = 0; i < n; i++)
        {
            if (!p[i])
                continue;
            return p[i];
        }
    }
    bool find(int x)
    {
        for (int i = n - 1; ~i; i--)
        {
            if ((x >> i) & 1)
                x ^= p[i];
        }
        if (!x)
            return 1;
        else
            return 0;
    }
    Linear_basis(int n = 64)
    {
        this->n = n;
        p.resize(n);
        p.clear();
    }
};
```

###### ST表

```cpp
//用于可重复贡献问题。
int num[MAXN];
int ST[MAXN][32];
void create() {
	for (int i = 0; i <= log2(N); i++) {
		for (int l = 1; l+(1<<i)-1<=N; l++) {
			if (!i) {
				ST[l][i] = num[l];
			}
			else {
				ST[l][i] = max(ST[l][i - 1], ST[l + (1<<(i-1))][i - 1]);//示例，这里可以改。此处写的是区间最大值问题。
			}
		}
	}
}
void output(int l, int r) {
	int k = log2(r-l+1);
	cout << max(ST[l][k], ST[r - (1 << k) + 1][k]) << endl;
}
```

###### 堆

```cpp
template<typename T>
struct SGheap {
	T Node[MAXN];
	int size;
	bool flag;//0-大根堆 1-小根堆
	void up(int ind) {
		if (ind / 2 && (flag ? Node[ind] > Node[ind / 2] :Node[ind] < Node[ind / 2])) {
			swap(Node[ind / 2], Node[ind]);
			up(ind / 2);
		}
	}
	T top() {
		return Node[1];
	}
	void down(int ind) {
		int t = ind;
		if (2 * ind <= size && (flag ? Node[2 * ind] > Node[t]:Node[2 * ind] < Node[t]))t = 2 * ind;
		if (2 * ind + 1 <= size && (flag ? Node[2 * ind + 1] > Node[t] : Node[2 * ind + 1] < Node[t]))t = 2 * ind + 1;
		if (t != ind) {
			swap(Node[ind], Node[t]);
			down(t);
		}
	}
	SGheap(bool _flag = 0) {
		size = 0;
		flag = _flag;
	}
	SGheap(T* A, int _size, bool _flag = 0) {
		size = _size;
		flag = _flag;
		for (int i = 1; i <= _size; i++) {
			Node[i] = A[i];
		}
		build_heap();
	}
	void build_heap() {//从下向上下沉可实现O(n)建堆
		for (int i = size; i >= 1; i--) {
			down(i);
		}
	}
	void pop() {
		Node[1] = Node[size--];
		down(1);
	}
	void insert(int x) {
		Node[++size] = x;
		up(size);
	}
};
//用法：
//(1)SGheap<int> Heap(0);小根堆
//(2)SGheap<int> Heap(num[],n,0)直接把数组传进去(下标从1开始)
```



###### 树状数组

```cpp
int sum[500010];
int c[500010];
int lowbit(int x) {
	return x & -x;
}
void create() {//利用前缀和O(n)建树
	for (int i = 1; i <= n; i++) {
		c[i] = sum[i] - sum[i - lowbit(i)];
	}
}
int getsum(int x) {
	int sum = 0;
	while (x) {
		sum += c[x];
		x = x - lowbit(x);
	}
	return sum;
}
void plusnum(int x, int k) {
	while (x <= n) {
		c[x] += k;
		x = x + lowbit(x);
	}
}
```

###### 线段树

```cpp
// 构造函数 SegmentTree SEGT(v)
// v下标从1开始
struct SegmentTree
{
    struct node
    {
        int val, lazy;
        node()
        {
            val = 0;
            lazy = 0;
        }
    };
    vector<node> v;
    int n;
    void pushdown(int id, int l, int r)
    {
        int mid = l + r >> 1;
        if (v[id].lazy && l != r)
        {
            int lz = v[id].lazy;
            v[id << 1].lazy += lz;
            v[id << 1].val += lz * (mid - l + 1);
            v[id << 1 | 1].lazy += lz;
            v[id << 1 | 1].val += lz * (r - mid);
            v[id].lazy = 0;
        }
    }
    void pushup(int id)
    {
        v[id].val = v[id << 1].val + v[id << 1 | 1].val;
    }
    void init(vector<int> &_v)
    {
        function<void(int, int, int)> buildtree = [&](int l, int r, int id)
        {
            if (l == r)
            {
                v[id].val = _v[l];
                return;
            }
            int mid = (l + r) >> 1;
            buildtree(l, mid, id << 1);
            buildtree(mid + 1, r, id << 1 | 1);
            pushup(id);
        };
        buildtree(1, n, 1);
    }
    void update(int L, int R, int l, int r, int id, int delta)
    {
        if (l >= L && r <= R)
        {
            v[id].lazy += delta;
            v[id].val += delta * (r - l + 1);
            return;
        }
        int mid = l + r >> 1;
        pushdown(id, l, r);
        if (L <= mid)
            update(L, R, l, mid, id << 1, delta);
        if (mid + 1 <= R)
            update(L, R, mid + 1, r, id << 1 | 1, delta);
        pushup(id);
    }
    void update(int L, int R, int delta)
    {
        update(L, R, 1, n, 1, delta);
    }
    int query(int L, int R, int l, int r, int id)
    {
        if (l >= L && r <= R)
        {
            return v[id].val;
        }
        int mid = l + r >> 1;
        pushdown(id, l, r);
        int sum = 0;
        if (L <= mid)
            sum += query(L, R, l, mid, id << 1);
        if (mid + 1 <= R)
            sum += query(L, R, mid + 1, r, id << 1 | 1);
        return sum;
    }
    int query(int L, int R)
    {
        return query(L, R, 1, n, 1);
    }
    SegmentTree(vector<int> &_v)
    {
        n = _v.size() - 1;
        v.resize((n << 2) + 5);
        init(_v);
    }
};
```

###### 单调队列

```cpp
//如果新加进来的值比原来队尾的大，则弹出队尾。
//能待的时间更长且更优的留在队列里，其余弹出。
int num[MAXN],q[MAXN],tail=0,head=-1;
void search(){
	for (int i = 0; i < n; i++) {//以最大值为例
		if (head <= tail && q[head] <= i - k) {
			head++;
		}
		while (head <= tail && a[q[tail]] <= a[i]) {
			tail--;
		}
		q[++tail] = i;
		if (i >= k - 1)//do something...
	}
}
```

###### 单调栈

```cpp
int num[MAXN],st[MAXN],top=-1;
void search(){
	for (int i = 0; i < n; i++) {
		while (top!=-1&&num[st[top]] <= num[i]) {
			top--;
		}
        //此时st[top]存储的是所需的下标.例子中是左侧离它最近的大于它的值的数的下标
		st[++top] = i;	
	}
}
```



# 动态规划

###### 状压dp

```cpp
//从状态 i 转移到状态 j	复杂度O(n^2*2^n)
_for(k, 0, (1 << n)) {
	_for(i, 0, n) {
		if (!(k & (1 << i)))continue;//如果在考虑的k中i（起点）没有出现，则跳过
		_for(j, 0, n) {
			if (i == j)continue;//如果在考虑的k中i==
			if (!(k & (1 << j)))continue;//如果在考虑的k中j（终点）没有出现，则跳过
			if (满足条件) {
				//dp[k][j]=....
			}
		}
	}
}
```

###### 概率dp

```cpp
//概率dp从初始状态开始，考虑某一个状态后的状态转移，比如原来是dp[3] 经过某一转化后(概率p)变成dp[1]，则dp[3]=dp[1]*p

//期望dp从末状态开始（期望为0），考虑这个状态是哪里来的，然后加上走这一步需要的代价。比如走一步要v，dp[4]可以由dp[3]转化而来(概率p),则dp[3]=dp[4]*p+v
```

###### 枚举子集

```cpp
for (int d = 0; d <= D; d++) {
        for (int i = 2; i <= n; i++) {
            if (i & (1 << d)) {
                dis[i] = min(dis[i], dis[i ^ (1 << d)] + i * k);
            }
        }
    }
```

###### 数位dp

example:windy数

求[l,r]中相邻两位数之差>=2的数

```cpp
vector<int> v;
vector<vector<vector<int>>> dp(20, vector<vector<int>>(10, vector<int>(2, -1)));
// pos:当前考虑位
// limit:是否贴上届
// lead:上一位是否是前导零
// pre:上一位数字是多少
int dfs(int pos, bool limit, bool lead, int pre)
{
    if (pos < 0)
        return 1;
    if (!limit && dp[pos][pre][lead] != -1)
    {
        return dp[pos][pre][lead];
    }
    int sup = limit ? v[pos] : 9;
    int ans = 0;
    for (int i = 0; i <= sup; i++)
    {
        if (!lead && (abs(i - pre) < 2))
            continue;
        ans += dfs(pos - 1, limit && i == sup, lead && i == 0, i);
    }
    if (!limit)
    {
        dp[pos][pre][lead] = ans;
    }
    return ans;
}
int cal(int x)
{
    if (x < 0)
        return 0;
    v.clear();
    while (x)
    {
        v.emplace_back(x % 10);
        x /= 10;
    }
    return dfs((int)v.size() - 1, 1, 1, 0);
}
```



# 图论

###### 拓扑排序

```cpp
// 返回拓扑序
vector<int> topoSort(vector<vector<int>> &g)
{
    queue<int> q;
    vector<int> ret;
    int n = g.size() - 1;
    vector<int> c(n + 1);
    for (int i = 1; i <= n; i++)
        for (int &to : g[i])
            c[to]++;
    for (int i = 1; i <= n; i++)
        if (c[i] == 0)
            q.emplace(i);
    while (!q.empty())
    {
        int u = q.front();
        ret.emplace_back(u);
        q.pop();
        for (int &to : g[u])
            if (--c[to] == 0)
                q.emplace(to);
    }
    return ret;
}
```



###### Dijkstra算法

```cpp
// 单源最短路（无负权边）
vector<int> dijkstra(vector<vector<pair<int, int>>> &g, int &s)
{
    int n = g.size() - 1;
    vector<int> dis(n + 1, -1);
    vector<bool> vis(n + 1);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
    dis[s] = 0;
    q.emplace(make_pair(0, s));
    while (!q.empty())
    {
        int u = q.top().second;
        q.pop();
        if (vis[u])
            continue;
        vis[u] = 1;
        for (auto &[to, w] : g[u])
        {
            if (dis[to] > dis[u] + w || dis[to] == -1)
            {
                dis[to] = dis[u] + w;
                q.emplace(make_pair(dis[to], to));
            }
        }
    }
    return dis;
}
```



###### SPFA

```cpp
// 单源最短路（有负权边）（无负环）
vector<int> SPFA(vector<vector<pair<int, int>>> &g, int &s)
{
    int n = g.size() - 1;
    queue<int> q;
    vector<int> dis(n + 1, inf);
    vector<bool> vis(n + 1); // 是否在队
    dis[s] = 0;
    q.emplace(s);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (auto &[to, w] : g[u])
            if (dis[to] > dis[u] + w)
            {
                dis[to] = dis[u] + w;
                if (!vis[to])
                {
                    vis[to] = 1;
                    q.emplace(to);
                }
            }
    }
    return dis;
}
```



###### 链式前向星

```cpp
int head[10010], tot = 1;
struct Node {
	int to;
	int next;
	int len;
}e[20010];
void create(int u,int to, int len) {
	e[tot].to = to;
	e[tot].len = len;
	e[tot].next = head[u];
	head[u] = tot++;
}
void search(int u) {
	for (int i = head[u]; i; i = e[i].next) {
		int to = e[i].to;
		int len = e[i].len;
		//.....
	}
}
```

###### 匈牙利算法

```cpp
struct Node {
	vector<int>v;
}a[MAXN];
int place[MAXN];
bool isvisited[MAXN];
bool find(int x) {
	for (auto i = a[x].v.begin(); i != a[x].v.end(); i++) {
		if (!isvisited[*i]) {
			++isvisited[*i];
			if (!place[*i] || find(place[*i])) {
				place[*i] = x;
				return 1;
			}
		}
	}
	return 0;
}
void km(){
	for (int i = 1; i <= n; i++) {
		memset(isvisited, 0, sizeof(isvisited));
		if (find(i))tot++;
	}
}
```

###### 网络最大流

```cpp
// 先用 max_flow(n,s,t)初始化，再用add(u,v,c)加边，最后直接输出work()
struct max_flow {
    int n;
    int s, t;
    vector<vector<int>> g;
    vector<pair<int, int>> e;
    vector<int> dis;
    vector<int> cur;
    max_flow();
    max_flow(int n, int s, int t) : n(n), g(n + 1), s(s), t(t) {}
    void add(int u, int v, int c) {
        g[u].emplace_back(e.size());
        e.emplace_back(v, c);
        g[v].emplace_back(e.size());
        e.emplace_back(u, 0);
    }
    bool bfs() {
        dis.assign(n + 1, -1);
        cur.assign(n + 1, 0);
        queue<int> q;
        q.emplace(s);
        dis[s] = 0;

        while (!q.empty()) {
            const int ind = q.front();
            q.pop();

            for (int &i : g[ind]) {
                auto [to, v] = e[i];

                if (v > 0 && dis[to] == -1) {
                    q.emplace(to);
                    dis[to] = dis[ind] + 1;

                    if (to == t)
                        return true;
                }
            }
        }

        return false;
    }
    int dfs(int ind, int flow) {
        if (ind == t) {
            return flow;
        }

        int rmn = 0;

        for (int i = cur[ind]; i != g[ind].size(); i++) {
            cur[ind] = i;
            const int j = g[ind][i];
            auto [to, v] = e[g[ind][i]];

            if (dis[to] == dis[ind] + 1) {
                int c = dfs(to, min(flow - rmn, v));
                e[j].second -= c;
                e[j ^ 1].second += c;
                rmn += c;

                if (rmn == flow)
                    return rmn;
            }
        }

        return rmn;
    }
    int work() {
        int ans = 0;

        while (bfs()) {
            ans += dfs(s, 1e18);
        }

        return ans;
    }
};
```



###### 费用流

```cpp
struct MinCostFlow
{
    using LL = long long;
    using PII = pair<LL, int>;
    const LL INF = numeric_limits<LL>::max();
    struct Edge
    {
        int v, c, f;
        Edge(int v, int c, int f) : v(v), c(c), f(f) {}
    };
    const int n;
    vector<Edge> e;
    vector<vector<int>> g;
    vector<LL> h, dis;
    vector<int> pre;
    MinCostFlow(int n) : n(n), g(n + 1) {}
    void add(int u, int v, int c, int f)
    { // c 流量, f 费用
        g[u].push_back(e.size());
        e.emplace_back(v, c, f);
        g[v].push_back(e.size());
        e.emplace_back(u, 0, -f);
    }
    bool dijkstra(int s, int t)
    {
        dis.assign(n + 1, INF);
        pre.assign(n + 1, -1);
        priority_queue<PII, vector<PII>, greater<PII>> que;
        dis[s] = 0;
        que.emplace(0, s);
        while (!que.empty())
        {
            auto [d, u] = que.top();
            que.pop();
            if (dis[u] < d)
                continue;
            for (int i : g[u])
            {
                auto [v, c, f] = e[i];
                if (c > 0 && dis[v] > d + h[u] - h[v] + f)
                {
                    dis[v] = d + h[u] - h[v] + f;
                    pre[v] = i;
                    que.emplace(dis[v], v);
                }
            }
        }
        return dis[t] != INF;
    }
    pair<int, LL> flow(int s, int t)
    {
        int flow = 0;
        LL cost = 0;
        h.assign(n + 1, 0);
        while (dijkstra(s, t))
        {
            for (int i = 1; i <= n; ++i)
                h[i] += dis[i];
            int aug = numeric_limits<int>::max();
            for (int i = t; i != s; i = e[pre[i] ^ 1].v)
                aug = min(aug, e[pre[i]].c);
            for (int i = t; i != s; i = e[pre[i] ^ 1].v)
            {
                e[pre[i]].c -= aug;
                e[pre[i] ^ 1].c += aug;
            }
            flow += aug;
            cost += LL(aug) * h[t];
        }
        return {flow, cost};
    }
};
```



# 数学

###### 数论分块

$\lfloor\frac n i\rfloor块的右端点在\lfloor\frac{n}{\lfloor\frac ni\rfloor}\rfloor，左端点在\lfloor\frac{n}{\lfloor\frac n{i-1}\rfloor}\rfloor+1$

$\lceil\frac ni\rceil块的右端点在\lfloor\frac{n-1}{\lfloor\frac{n-1}i\rfloor}\rfloor，左端点在\lfloor\frac{n-1}{\lfloor\frac{n-1}{i-1}\rfloor}\rfloor+1$

```cpp
//省流:值n/i的右端点在n/(n/i)
//例:
ll fenkuai(ll n) {//枚举i，将n/i相加。
	int ans = 0;
	int l = 1, r;
	while (l<=n) {
		r = n / (n / l);
		ans += (r - l + 1) * (n / l);
		l = r + 1;
	}
	return ans;
}
```

###### 高斯消元

```cpp
//用于求解线性方程组
//传入double类型增广矩阵，直接返回解
//若无解或解不唯一则返回空vector
vector<double> Gaussian_elimination(vector<vector<double>> v)
{
    int n = v.size();
    for (int i = 0; i < n; i++)
    {
        // 选定的行
        int maxm = 0;
        int maxind = -1;
        // 寻找在这一列有有效值的行
        for (int j = i; j < n; j++)
        {
            if (abs(v[j][i]) > maxm)
            {
                maxm = abs(v[j][i]);
                maxind = j;
            }
        }
        if (maxind == -1)
        {
            return vector<double>();
        }
        swap(v[i], v[maxind]);
        // 归一化
        double d = 1 / v[i][i];
        for (int j = i; j < n + 1; j++)
        {
            v[i][j] *= d;
        }
        // 将其他行对应该列元素消掉
        for (int j = i + 1; j < n; j++)
        {
            if (v[j][i] == 0)
                continue;
            double d1 = v[j][i];
            for (int t = i; t < n + 1; t++)
            {
                v[j][t] -= d1 * v[i][t];
            }
        }
    }
    vector<double> ans(n);
    for (int i = n - 1; i >= 0; i--)
    {
        ans[i] = v[i][n] / v[i][i];
        // 反向向上消元
        for (int j = i - 1; j >= 0; j--)
        {
            if (v[j][i] == 0)
                continue;
            double d1 = v[j][i];
            for (int t = i; t < n + 1; t++)
            {
                v[j][t] -= d1 * v[i][t];
            }
        }
    }
    return ans;
}
```



###### MillerRabin & Pollard Rho

```cpp
struct MillerRabin {
    vector<int> Prime;
    MillerRabin() :Prime({ 2,3,5,7,11,13,17,19,23 }) {}
    static constexpr int mulp(const int& a, const int& b, const int& P) {
        int res = a * b - (int)(1.L * a * b / P) * P;
        res %= P;
        res += (res < 0 ? P : 0);
        return res;
    }
    static constexpr int powp(int a, int mi, const int& mod) {
        int ans = 1;
        for (; mi; mi >>= 1) {
            if (mi & 1) ans = mulp(ans, a, mod);
            a = mulp(a, a, mod);
        }
        return ans;
    }
    bool operator()(const int& v) {
        if (v < 2 || v != 2 && v % 2 == 0) return false;
        int s = v - 1;
        while (!(s & 1)) s >>= 1;
        for (int x : Prime) {
            if (v == x) return true;
            int t = s, m = powp(x, s, v);
            while (t != v - 1 && m != 1 && m != v - 1) m = mulp(m, m, v), t <<= 1;
            if (m != v - 1 && !(t & 1))return false;
        }
        return true;
    }
};
struct PollardRho :public MillerRabin {
    mt19937 myrand;
    PollardRho() :myrand(time(0)) {}
    int rd(int l, int r) {
        return myrand() % (r - l + 1) + l;
    }
    int operator()(int n) {
        if (n == 4) return 2;
        MillerRabin& super = *this;
        if (super(n)) return n;
        while (1) {
            int c = rd(1, n - 1);
            auto f = [&](int x) {
                return (super.mulp(x, x, n) + c) % n;
                };
            int t = 0, r = 0, p = 1, q;
            do {
                for (int i = 0; i < 128; i++) {
                    t = f(t), r = f(f(r));
                    if (t == r || (q = super.mulp(p, abs(t - r), n)) == 0) break;
                    p = q;
                }
                int d = gcd(p, n);
                if (d > 1) return d;
            } while (t != r);
        }
    }
};
```



###### 计算几何

```cpp
struct Point{
    double x,y;
    double operator*(const Point &e) const{
        return x*e.x+y*e.y;
    };
    Point operator*(const double k) const{
        return {x*k,y*k};
    }
    double operator^(const Point &e) const{
        return x*e.y-e.x*y;   
    }
    Point operator+(const Point &e) const{
        return {x+e.x,y+e.y};
    }
    Point operator-(const Point &e) const{
        return {x-e.x,y-e.y};
    }
    Point operator/(const double &k) const{
        return {x/k,y/k};
    }
    //象限
    inline int quad() const{
        if(x>0&&y>=0) return 1;
        if(x<=0&&y>0) return 2;
        if(x<0&&y<=0) return 3;
        if(x>=0&&y<0) return 4;
        return 5;
    }
    inline static bool sortxupyup(const Point &a,const Point &b){
        if(a.x!=b.x) return a.x<b.x;
        else return a.y<b.y;
    }
    //极角排序
    inline static bool sortPointAngle(const Point &a,const Point &b){
        if(a.quad()!=b.quad()) return a.quad()<b.quad();
        return (a^b)>0;
    }
    //模长
    inline double norm() const{
        return sqrtl(x*x+y*y);
    }
    //向量方向
    //1 a在b逆时针方向
    //0 同向或反向
    //2 a在b顺时针方向
    int ordervector(const Point &e){
        double p=(*this)^e;
        if(p>0) return 1;
        else if(p==0.0) return 0;
        else return 2;
    }
    //逆时针旋转alpha角
    inline Point Spin(double alpha){
        double sinalpha=sin(alpha);
        double cosinalpha=cos(alpha);
        return {x*cosinalpha-y*sinalpha,x*sinalpha+y*cosinalpha};
    }
    inline double dis(const Point &e){
        Point c=(*this)-e;
        return c.norm();
    }
    double getangle(const Point &e) const{
        return fabs(atan2l(*this^e,*this*e));
    }
};
struct Line{
    //过x点，方向向量为y
    Point x,y;
    //type=0,点和方向向量
    //type=1，点和点
    Line(const Point &a,const Point &b,int type){
        if(type==0){
            x=a,y=b;
        }else{
            x=a;
            y=b-a;
        }
    }
    inline double distancetopoint(const Point &e) const{
        return fabs((e-x)^y)/y.norm();
    }
};
//要先getConvex求凸包，其他的才能用
struct Polygon{
    vector<Point> p;
    vector<Point> convexhull;
    int n;
    Polygon(int n,vector<Point> &v):n(n),p(v){} 
    Polygon(int n):n(n),p(n){}
    void input(){
        for(int i=0;i<n;i++){
            cin>>p[i].x>>p[i].y;
        }
    }
    void getConvex(){
        sort(p.begin(),p.end(),Point::sortxupyup);
        p.erase(unique(p.begin(),p.end(),[](const Point &a,const Point &b){
            return a.x==b.x&&a.y==b.y;
        }),p.end());
        n=p.size();
        if(n==0) return;
        if(n==1){
            convexhull.push_back(p.front());
            convexhull.push_back(p.front());
            return;
        }
        vector<int> st(2*n+5,0);
        vector<bool> used(n,0);
        int tp=0;
        st[++tp]=0;
        for(int i=1;i<n;i++){
            while(tp>=2&&((p[st[tp]]-p[st[tp-1]])^(p[i]-p[st[tp]]))<=0){
                used[st[tp--]]=0;
            }
            used[i]=1;
            st[++tp]=i;
        }
        int tmp=tp;//下凸壳大小
        for(int i=n-2;i>=0;i--){
            if(!used[i]){
                while(tp>tmp&&((p[st[tp]]-p[st[tp-1]])^(p[i]-p[st[tp]]))<=0){
                    used[st[tp--]]=0;
                }
                used[i]=1;
                st[++tp]=i;
            }
        }
        for(int i=1;i<=tp;i++){
            convexhull.push_back(p[st[i]]);
        }
    }
    double getPerimeter(){
        double ans=0;
        for(int i=1;i<convexhull.size();i++){
            ans+=convexhull[i].dis(convexhull[i-1]);
        }
        return ans;
    }
    double getArea(){
        if(convexhull.size()<4) return 0;
        double ans=0;
        for(int i=1;i<convexhull.size()-2;i++){
            ans+=(convexhull[i]-convexhull[0])^(convexhull[i+1]-convexhull[0])/2;
        }
        return ans;
    }
    //旋转卡壳求直径
    double getLongest(){
        if(convexhull.size()<4){
            return convexhull[0].dis(convexhull[1]);
        }
        int j=0;
        const int sz=convexhull.size();
        double ans=0;
        for(int i=0;i<convexhull.size()-1;i++){
            Line line(convexhull[i],convexhull[i+1],1);
            while(line.distancetopoint(convexhull[j])<=line.distancetopoint(convexhull[(j+1)%sz])){
                j=(j+1)%sz;
            }
            ans=max({ans,(convexhull[i]-convexhull[j]).norm(),(convexhull[i+1]-convexhull[j]).norm()});
        }
        return ans;
    }
    //旋转卡壳最小矩形覆盖
    pair<double,vector<Point>> minRectangleCover(){
        vector<Point> p;
        if(convexhull.size()<4) return {0,p};
        int j=1,l=1,r=1;
        double ans=1e18;
        const int sz=convexhull.size();
        for(int i=1;i<convexhull.size();i++){
            Line line(convexhull[i-1],convexhull[i],1);
            while(line.distancetopoint(convexhull[j])<=line.distancetopoint(convexhull[(j+1)%sz])){
                j=(j+1)%sz;
            }
            while((convexhull[i]-convexhull[i-1])*(convexhull[(r+1)%sz]-convexhull[i-1])>=(convexhull[i]-convexhull[i-1])*(convexhull[r]-convexhull[i-1])){
                r=(r+1)%sz;
            }
            if(i==1) l=r;
            while((convexhull[i-1]-convexhull[i])*(convexhull[(l+1)%sz]-convexhull[i])>=(convexhull[i-1]-convexhull[i])*(convexhull[l]-convexhull[i])){
                l=(l+1)%sz;
            }
            Point t1=convexhull[i]-convexhull[i-1];
            Point t2=convexhull[r]-convexhull[i];
            Point t3=convexhull[l]-convexhull[i-1];
            double a=line.distancetopoint(convexhull[j]);
            double b=t1.norm()+t1*t2/t1.norm()-t1*t3/t1.norm();
            double tmp=a*b;
            if(ans>tmp){
                ans=tmp;
                p.clear();
                p.push_back(t1*((t1*t3)/(t1.norm()*t1.norm()))+convexhull[i-1]);
                p.push_back(t1*(1+(t1*t2)/(t1.norm()*t1.norm()))+convexhull[i-1]);
                Point tmp=Point{-(p[1]-p[0]).y,(p[1]-p[0]).x}*a/b;
                p.push_back(tmp+p[1]);
                p.push_back(tmp+p[0]);
            }
        }
        return {ans,p};
    }
};
```



###### 三点求一圆解析

```cpp
vector<double> get_circle(double x1, double y1, double x2, double y2, double x3, double y3)
{
    double a = x1 - x2;
    double b = y1 - y2;
    double c = x1 - x3;
    double d = y1 - y3;
    double e = ((x1 * x1 - x2 * x2) - (y2 * y2 - y1 * y1)) / 2;
    double f = ((x1 * x1 - x3 * x3) - (y3 * y3 - y1 * y1)) / 2;

    // 圆心位置
    double x = (e * d - b * f) / (a * d - b * c);
    double y = (a * f - e * c) / (a * d - b * c);
    double r = sqrt(sqr(x1 - x) + sqr(y1 - y));
    vector<double> v;
    v.emplace_back(x);
    v.emplace_back(y);
    v.emplace_back(r);
    return v;
}
```



###### 康托展开

```cpp
ll n;
ll num[MAXN];
ll tree[MAXN];
ll jiecheng[MAXN];
void jc(){
	jiecheng[0] = 1;
	_for(i, 1, MAXN) {
		jiecheng[i] = jiecheng[i - 1] * i % mod;
	}
}
ll lowbit(ll x) {
	return x & -x;
}
ll query(ll x) {
	ll sum = 0;
	while (x) {
		sum += tree[x];
		x -= lowbit(x);
	}
	return sum;
}
void update(ll x, ll k) {
	while (x <= n) {
		tree[x] += k;
		x = x + lowbit(x);
	}
}
ll cantor() {
	ll ans = 1;
	jc();
	_for(i, 0, n) {
		update(num[i], 1);
		ans += (num[i] - query(num[i])) * jiecheng[n-i-1];
		ans %= mod;
	}
	return ans % mod;
}
```



###### 线性筛

```cpp
int prime[MAXN];
int sz;
bool visited[MAXN];
for (int i = 2; i <= MAXN; i++) {
	if (!visited[i]) prime[sz++] = i;
	for (int j = 0; i*prime[j]<=MAXN&&j < sz; j++) {
		visited[prime[j] * i]++;
		if (i % prime[j] == 0)break;
	}
}
```



###### 矩阵

```cpp
struct Matrix {
	ll a[2][2];
	Matrix() {
		memset(a, 0, sizeof(a));
	}
	Matrix operator*(const Matrix& other)const {
		Matrix res;
		for (int i = 0; i < 2; i++) 
			for (int j = 0; j < 2; j++) 
				for (int d = 0; d < 2; d++) 
					res.a[i][j] = res.a[i][j]+a[i][d]*other.a[d][j];
		return res;
	}
	Matrix operator%(int mod)const {
		Matrix res;
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				res.a[i][j] = a[i][j] % mod;
		return res;
	}
}bit;
//快速幂
Matrix fastpow(Matrix a, int b, int mod) {
	Matrix ans = a;
	while (b) {
		if (b & 1)ans = ans * a % mod;
		a = a * a % mod;
		b >>= 1;
	}
	return ans;
}
//加速递推斐波那契
void init() {
	bit.a[0][0] = bit.a[1][0] = bit.a[0][1] = 1;
	bit.a[1][1] = 0;
}
```



###### 欧几里得

```cpp
int gcd(int a,int b){
    return(b?gcd(b,a%b):a);
}
```

###### 扩展欧几里得

```cpp
int exgcd(int a, int b,int& x,int& y) {
	if (b == 0) {
		x = 1;
		y = 0;
		return a;
	}
	int d = exgcd(b, a % b, x, y);
	int t = x;
	x = y;
	y = t - a / b * x;
	return d;
}
int frac(int p, int q, int mod) {//exgcd求p/q%mod
	int x = 1, y = 1;
	exgcd(q, mod, x, y);
	return (p * (x + mod) % mod) % mod;
}
```

###### 快速幂

```cpp
int qpow(int a, int b,int mod) {
	int ans = 1;
	while (b) {
		if(b & 1)ans = ans * a % mod;
		a = a * a % mod;
		b >>= 1;
	}
	return ans;
}
int frac(int p, int q,int mod) {//费马小定理求p/q%mod
	return p * qpow(q, mod-2, mod) % mod;
}
```

###### 排列组合exlucas

```cpp

```

###### 调和级数

$\Sigma^n_{i=1}\frac1i=\ln(n+1)+\gamma$

$\gamma=0.5772156649$

# 字符串

###### 字符串哈希

```cpp
const int HASHMOD[2] = { 998244353,(int)1e9 + 7 };
const int BASE[2] = { 29,31 };
struct Stringhash {
    static vector<int> qpow[2];
    vector<int> hash[2];
    void init() {
        qpow[0].push_back(1);
        qpow[1].push_back(1);
        for (int i = 1; i <= 1e6; i++) {
            for (int j = 0; j < 2; j++) {
                qpow[j].push_back(qpow[j].back() * BASE[j] % HASHMOD[j]);
            }
        }
    }
    Stringhash(string& s, int base) {
        for (int i = 0; i < 2; i++) {
            hash[i] = vector<int>(s.size() + 1);
            hash[i][0] = 0;
        }
        if (qpow[0].empty()) init();
        for (int i = 1; i <= s.size(); i++) {
            for (int j = 0; j < 2; j++) {
                hash[j][i] = (hash[j][i - 1] * BASE[j] % HASHMOD[j] + s[i - 1] - base) % HASHMOD[j];
            }
        }
    }
    pair<int, int> gethash(int x, int y) {
        pair<int, int> result = { 0,0 };
        for (int i = 0; i < 2; i++) {
            int k = ((hash[i][y] - hash[i][x - 1] * qpow[i][y - x + 1]) % HASHMOD[i] + HASHMOD[i]) % HASHMOD[i];
            if (i == 0) result.first = k;
            else result.second = k;
        }
        return result;
    }
};
vector<int> Stringhash::qpow[2];
```

###### 字典树

```cpp
int tr[MAXN][26],cnt,ed[MAXN];
bool find(string str) {//查找字符串
	int p = 0;
	for (int i = 0; i < str.size(); i++) {
		int ch = str[i] - 'a';
		if (!tr[p][ch]) return 0;
		p = tr[p][ch];
	}
    return ed[p];
}
void insert(string str) {//插入字符串
	int p = 0;
	for (int j = 0; j < str.length(); j++) {
		int ch = str[j] - 'a';
		if (!tr[p][ch])tr[p][ch] = ++cnt;//若子节点并不存在，则新建
		p = tr[p][ch];
	}
	ed[p] = 1;//记录某个串的结尾
}
```

###### 子序列自动机

```cpp
//从前往后找子序列的结尾
int nxt[MAXN][MAXM];
string str;
void init_nxt(){
    for (int i = n; i > 0; i--) {
		if (i != n)
			for (int j = 0; j < MAXN; j++) {
				nxt[i][j] = nxt[i + 1][j];
			}
		else
			for (int j = 0; j < MAXN; j++) {
				nxt[i][j] = inf;
			}
		if (str[i-1] >= 'A' && str[i-1] <= 'Z') {//要改
			nxt[i][str[i-1] - 'A'] = i;//要改
		}
	}
}
int query_nxt(){
    int temp= 0;
    for (int j = 0; j < str.size(); j++) {
		temp++;
		if (temp > n) {
			return n+1;
		}
		temp = nxt[temp][str[j] - 'A'];//要改
	}
    return temp;
}
//从后往前找子序列的开头
int pre[MAXN][MAXM];
string str;
void init_pre(){
    for (int i = 1; i <= n; i++) {
		if (i != 1)
			for (int j = 0; j < MAXN; j++) {
				pre[i][j] = pre[i - 1][j];
			}
		else {
			for (int j = 0; j < MAXN; j++) {
				pre[i][j] = 0;
			}
		}
		if (str[i-1] >= 'A' && str[i-1] <= 'Z') {//要改
			pre[i][str[i-1] - 'A'] = i;//要改
		}
	}
}
int query_pre(){
    int temp= n + 1;
    for (int j = 0; j < str.size(); j++) {
		temp--;
		if (temp < 1) {
			return 0;
		}
		temp = nxt[temp][str[j] - 'A'];//要改
	}
    return temp;
}
```



###### KMP

```cpp
int nxt[MAXN];
void kmpcheck(string txt,string str) {
	int i = 0, j = 0;
	while (i != str.length()) {
		if (str[i] == txt[j]) {
			i++;
			j++;
		}
		else if (str[i]!=txt[j]) {
			if (j > 0)j = nxt[j - 1];
			else i++;
		}
		if (j==txt.length()) {
			//.
			j = nxt[j - 1];
		}
	}
}
void create_next(string txt) {//构建模式串的broder
	for (int i = 1; i < txt.length(); i++) {
		int j = nxt[i - 1];
		while (j && txt[i] != txt[j]) {
			j = nxt[j - 1];
		}
		if (txt[i] == txt[j])nxt[i] = j + 1;
        else nxt[i] = 0;
	} 
}
```
