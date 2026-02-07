# 📝 论文写作指南

## Method Section 写作要点

### 3.1 Problem Formulation

```
Given a time series classification model f and an input x ∈ R^(C×T), 
we aim to provide RELIABLE explanations that identify which timesteps 
are truly influential for the prediction.

Unlike existing methods that focus solely on attribution IMPORTANCE, 
our approach explicitly models explanation UNCERTAINTY and CONSISTENCY 
to assess RELIABILITY.
```

**关键点：**
- 明确说明关注reliability而不只是importance
- 引入uncertainty和consistency概念

---

### 3.2 Multi-view Attribution with Uncertainty

```
We compute attributions using Integrated Gradients with MC Dropout 
to quantify explanation uncertainty:

A_t^μ = (1/M) Σ IG(x, c)_t^(m)
A_t^σ = std({IG(x, c)_t^(m)}_{m=1}^M)

where M is the number of MC samples. High σ indicates the attribution 
at timestep t is UNSTABLE across stochastic forward passes.
```

**关键点：**
- 强调uncertainty反映不稳定性
- 不是说uncertainty高就不重要，而是不可靠

---

### 3.3 Time-step-level Cross-view Consistency

**❌ 错误写法（Global）：**
```
We compute consistency as the cosine similarity between attributions 
from different views:
C = cos(A_view1, A_view2)
```

**✅ 正确写法（Time-step-level）：**
```
We compute consistency at EACH TIMESTEP by measuring the agreement 
across views:

C_t = 1 / (1 + σ({A_t^(v)}_{v=1}^V))

where {A_t^(v)} are attribution values at timestep t across V views.
This produces a consistency score for EACH timestep, rather than a 
single global value.
```

**关键点：**
- 明确说明是time-step-level
- 解释为什么这比global更好（可以看到哪些时间步一致）

---

### 3.4 Trust Score: Reliability, Not Importance

**核心定义：**
```
The trust score measures explanation RELIABILITY:

T_t = α(1 - σ_t^norm) + β C_t

where:
- σ_t^norm is normalized uncertainty at timestep t
- C_t is time-step-level consistency
- α, β are weights (default 0.5)

IMPORTANT: T_t is INDEPENDENT of attribution magnitude |A_t|.
A timestep can have:
1. High |A_t| but low T_t → important but UNRELIABLE
2. Low |A_t| but high T_t → unimportant but RELIABLE
3. High |A_t| and high T_t → important AND RELIABLE ✓
```

**关键点：**
- 明确trust独立于importance
- 用具体例子说明四种组合
- 强调高trust才是真正可信的重要点

---

### 3.5 Trust-aware Explanation

```
Rather than simply weighting by trust (which would mix importance 
and reliability), we use trust to FILTER unreliable attributions:

A_t^trusted = A_t · 𝟙(T_t > τ)

where τ is a trust threshold. This preserves the IMPORTANCE RANKING 
of reliable timesteps while DOWN-WEIGHTING unreliable ones.
```

**关键点：**
- 解释为什么不直接用 A_t × T_t（会混淆概念）
- 强调trust用于过滤，不改变重要性排序

---

## Experiments Section 写作要点

### 4.1 Experimental Setup

```
Datasets: We evaluate on X UCR datasets (ECG200, GunPoint, ...)
Models: SimpleResNet, Transformer, ...
Baselines: 
  - Original IG (no uncertainty)
  - Random attribution
  - Global consistency (view-level, not timestep-level)

Metrics:
  - Deletion AUC (explanation quality)
  - Prediction stability under noise (reliability)
  - Time-step consistency (not global similarity)
```

---

### 4.2 Explanation Uncertainty Analysis

**目标：** 证明uncertainty不是噪声，而是反映不可靠性

**写法：**
```
Figure X shows attribution mean and standard deviation on sample Y.
We observe three patterns:

1. High-attribution, low-uncertainty regions (e.g., timesteps 40-60)
   → Model is CONFIDENT about these timesteps' importance
   
2. High-attribution, high-uncertainty regions (e.g., timesteps 20-30)
   → Model is UNCERTAIN, suggesting these attributions are UNRELIABLE
   
3. Low-attribution regions show varying uncertainty levels
   → Uncertainty is NOT simply correlated with attribution magnitude

This demonstrates that uncertainty captures RELIABILITY independent 
of IMPORTANCE.
```

**图表：**
- 信号 + attribution mean + uncertainty bands
- 标注三类区域

---

### 4.3 Cross-view Consistency Evaluation

**目标：** 证明time-step-level consistency更好

**写法：**
```
Table X compares global view-level vs. time-step-level consistency:

Method                  | ECG200 | GunPoint | Avg
------------------------|--------|----------|-----
Global Cosine Sim       | 0.73   | 0.68     | 0.71
Timestep-level (ours)   | varies per timestep    

Figure Y shows time-step-level consistency for sample Z.
Key observations:

1. Decision-critical regions (e.g., peaks at t=50) show HIGH 
   consistency (C_t > 0.8) across views
   
2. Non-critical regions show LOWER consistency (C_t < 0.6)
   
3. Global similarity (0.71) MASKS this variation, losing important 
   information about WHICH timesteps are reliable

This validates our time-step-level approach.
```

**关键点：**
- 对比global和time-step-level
- 用具体例子说明time-step-level的优势
- 图表显示per-timestep的consistency变化

---

### 4.4 Trust-aware Explanation Evaluation

#### 实验A：四类时间步分析

```
We categorize timesteps by importance and reliability:

Category                    | Count | Avg Importance | Avg Trust
----------------------------|-------|----------------|----------
Reliable & Important       | 15    | 0.82          | 0.78
Unreliable & Important     | 8     | 0.79          | 0.32
Reliable & Unimportant     | 42    | 0.15          | 0.71
Unreliable & Unimportant   | 63    | 0.12          | 0.28

Observation: Category 2 (high importance, low trust) represents 
SPURIOUS attributions that APPEAR important but are UNRELIABLE.
Traditional methods would treat these as critical, leading to 
incorrect conclusions.
```

#### 实验B：Trust稳定性验证（关键！）

```
To verify that trust reflects RELIABILITY rather than IMPORTANCE, 
we conduct a perturbation experiment:

Setup:
- Select two groups with SIMILAR importance (|A| > 0.7):
  Group A: High trust (T > 0.7)  
  Group B: Low trust (T < 0.4)
  
- Add Gaussian noise (σ ∈ {0.1, 0.2, 0.3}) to these timesteps
- Measure prediction stability

Results (Table X):
                        | Pred Stability | Confidence Drop
------------------------|----------------|----------------
Group A (High Trust)    | 87.3%         | 0.08
Group B (Low Trust)     | 52.1%         | 0.31

Figure X shows these results visually.

Conclusion: Even with EQUAL attribution importance, high-trust 
timesteps remain STABLE under perturbation while low-trust 
timesteps are SENSITIVE to noise. This confirms that trust 
captures RELIABILITY independent of IMPORTANCE.
```

**关键点：**
- 这是最重要的实验！
- 证明trust ≠ importance
- 用客观数值展示差异

#### 实验C：Deletion实验

```
Table Y shows deletion AUC scores:

Method              | ECG200 | GunPoint | Avg
--------------------|--------|----------|-----
Original            | 0.62   | 0.58     | 0.60
Trust-weighted      | 0.71   | 0.68     | 0.70
Random              | 0.45   | 0.42     | 0.44

Trust-weighted explanations achieve HIGHER deletion AUC, indicating 
that removing high-trust timesteps causes FASTER prediction degradation.

This demonstrates that trust successfully identifies TRULY INFLUENTIAL 
and RELIABLE timesteps.
```

---

### 4.5 Ablation Study

```
Table Z shows ablation results:

Components              | Deletion AUC | Stability
------------------------|--------------|----------
Only Uncertainty        | 0.65         | 78.2%
Only Consistency        | 0.67         | 81.5%
Full (Uncertainty + C)  | 0.71         | 87.3%
No Trust (Baseline)     | 0.60         | 52.1%

Both components are necessary for optimal performance.
```

---

## Discussion Section 关键点

### 明确trust的角色

```
Our trust score serves as a META-MEASURE of explanation quality, 
answering "how reliable is this attribution?" rather than "how 
important is this timestep?". 

This distinction is crucial: traditional saliency methods answer 
WHAT is important, while our trust mechanism answers WHETHER we 
should BELIEVE those importance estimates.
```

### 实际应用价值

```
In high-stakes domains (healthcare, finance), it is insufficient 
to know WHAT the model considers important. We must also know 
WHETHER those attributions are TRUSTWORTHY.

For example, in ECG classification, a high-attribution region 
with low trust may indicate:
1. Model uncertainty about that region's role
2. Inconsistent explanations across different signal views
3. Potential for SPURIOUS correlations

Such regions warrant ADDITIONAL SCRUTINY before clinical use.
```

---

## Common Pitfalls to Avoid

**❌ 错误1：混淆trust和importance**
```
"Trust score identifies the most important timesteps."
```

**✅ 正确：**
```
"Trust score assesses the reliability of attributions. 
Important timesteps (high |A|) with low trust are potentially 
spurious and require caution."
```

---

**❌ 错误2：说global consistency**
```
"We measure consistency between views using cosine similarity."
```

**✅ 正确：**
```
"We compute consistency at each timestep by measuring cross-view 
agreement, producing a per-timestep reliability score rather than 
a single global measure."
```

---

**❌ 错误3：没有验证trust的独立性**
```
"High trust indicates important timesteps."
```

**✅ 正确：**
```
"Through perturbation experiments (Sec 4.4), we show that trust 
is independent of attribution magnitude: high-trust timesteps 
remain stable under noise regardless of their importance."
```

---

## 图表建议

### 必须有的图：

1. **Figure 1**: Method overview
   - Input → Multi-view decomposition → Attribution + Uncertainty
   - Time-step-level consistency → Trust score

2. **Figure 2**: Uncertainty analysis (Exp 1)
   - Signal + Attribution mean + Uncertainty bands
   - 标注三类区域

3. **Figure 3**: Time-step consistency (Exp 2)
   - Per-timestep consistency曲线
   - 对比关键/非关键区域

4. **Figure 4**: Four categories (Exp 3)
   - 2D scatter: importance vs trust
   - 四个象限清晰可见

5. **Figure 5**: Stability experiment (Exp 4) ⭐最重要
   - Bar chart: High-trust vs Low-trust stability
   - 清晰展示trust的意义

6. **Figure 6**: Deletion curves
   - Original vs Trust-weighted vs Random
   - Trust-weighted下降最快

### 表格建议：

- Table 1: 数据集统计
- Table 2: Time-step consistency对比
- Table 3: 四类时间步统计
- Table 4: 稳定性实验结果 ⭐
- Table 5: Deletion AUC对比
- Table 6: Ablation study

---

## 总结

**核心贡献的表述：**

```
We make three key contributions:

1. We introduce EXPLANATION UNCERTAINTY via MC Dropout to quantify 
   the stability of attributions.

2. We propose TIME-STEP-LEVEL cross-view consistency, moving beyond 
   global view-level measures to identify WHICH timesteps have 
   reliable explanations.

3. We define TRUST as a measure of explanation RELIABILITY independent 
   of attribution IMPORTANCE, and validate through perturbation 
   experiments that high-trust attributions are indeed more STABLE 
   and ROBUST.
```

祝论文写作顺利！📝
