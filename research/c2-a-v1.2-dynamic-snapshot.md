# C2-A v1.2：开盘累计完整分钟快照重算版

## 0. 定位与状态

本文档是 C2-A v1 的增量修订；未明确替换的 Universe、涨跌停、T+1、成交容量、冷却期和费用规则继续沿用 v1。

当前状态：`PAPER_ONLY`。本版本尚未完成样本外回测和前向纸面验证，不生成实盘权限。

## 1. 修改目标

解决两个问题：

1. 不再只使用开盘后固定30分钟的截面；手工扫描时，以最近一个已经完成的连续竞价分钟为快照时点，首根为 09:31。
2. 错过历史触发后不允许按当前市价追单，只能等待新的、未过期的信号。

## 2. 时间定义

```text
evaluation_time = 执行扫描时间
as_of = evaluation_time 之前最近一个已完成分钟
elapsed_minutes = 09:31 至 as_of 的完整连续竞价分钟数
```

任何指标只能使用 `as_of` 及之前已经存在的数据。正在进行中的分钟禁止使用。

为保留“早盘异常强势”假设，信号截止时间不直接拟合为单一最优值，而是预注册比较：

```text
SCAN_END ∈ {10:00, 10:30, 11:00}
```

当前 v1.2 前向纸面组使用 `11:00`；v1 的 `10:00` 作为必须保留的基准组。

## 3. 同长度历史基线

对每只股票、每个 `as_of` 分别计算：

```text
CumAmount(as_of)
= 当日 09:31 至 as_of 的累计成交额

AmountBurst(as_of)
= CumAmount(as_of)
  / 过去20个交易日在相同 elapsed_minutes 下的 CumAmount 中位数

CumVolume(as_of)
= 当日 09:31 至 as_of 的累计成交量

RelativeTurnover(as_of)
= CumVolume(as_of)
  / 过去20个交易日在相同 elapsed_minutes 下的 CumVolume 中位数
```

`RelativeTurnover` 代替原始换手率进入综合评分，降低小流通盘股票因天然换手高获得的结构性优势。

## 4. 数据质量门槛

严格信号必须同时满足：

- 当日和过去20日都有对应的完整一分钟数据；
- 同板块横截面覆盖当时全部 Universe 合格股票；
- 流通股本、流通市值、停牌、ST、上市天数和涨跌停状态可核验；
- 不得把5分钟数据、涨幅榜前N名或少量候选股的百分位冒充严格全市场c6。

任一条不满足：

```text
data_status = PROXY
execution_permission = PAPER_ONLY
```

## 5. 动态 c6_A

每个 `as_of` 分钟，在主板池和成长池内分别计算：

```text
P_amount = AmountBurst 横截面百分位
P_relative_turnover = RelativeTurnover 横截面百分位
P_gain = Gain 横截面百分位

Q = 0.50 * P_amount
  + 0.30 * P_relative_turnover
  + 0.20 * P_gain

c6_A = 100 * (1 - Q)
```

准入门槛仍为 `c6_A < 30`，但必须在 `as_of-1` 和 `as_of` 两个连续完整分钟同时满足 Universe、Gain 和 c6_A 条件。

## 6. 信号新鲜度与重置

首次连续两分钟满足条件时生成 `signal_id`。

```text
signal_expiry = signal_time + 3个完整分钟
```

信号到期、或回撤触发时应交却未申报/未成交时，状态改为：

```text
MISSED_ENTRY
```

`MISSED_ENTRY` 禁止按当前市价追单。只有先连续两分钟出现以下任一重置条件，才能重新激活：

- `c6_A >= 35`；
- Gain 离开所在板块的允许区间。

重置后再次连续两分钟满足全部条件，生成新的 `signal_id`。

## 7. 快照入场门槛

信号新鲜时，使用截至上一完整分钟的滚动最高价 `H_prev`。

继续使用 v1 的首仓回撤：

```text
主板 trigger = H_prev * 0.970
创业板/科创板 trigger = H_prev * 0.955
```

最新完整分钟必须同时满足：

- `minute_low <= trigger`；
- 最新价没有比 trigger 超调过深：主板不超过1%，成长板不超过1.5%；
- 不处于一字涨停、一字跌停或无可成交对手盘状态；
- 模拟成交额不超过该分钟成交额的5%；
- 满足价格笼子、价格最小变动单位和交易数量要求。

## 8. 小规模资金规则

对于10万元总资金的验证账户：

- 未晋级前，当日策略总风险暴露不超过1万元；
- 单股总成本不超过5,000元；
- 最多两只股票，禁止同股补仓；
- 普通股票按100股整数倍，科创板按最低200股及现行规则处理；
- 因整手、佣金或单股上限导致无法用满1万元时，剩余资金保留现金。

## 9. 交易后状态

买入后继续沿用 v1：

```text
D日：禁止主动卖出
D+1：原则上开盘退出
一字跌停：LOCKED_LIMIT_DOWN
亏损标的：COOLDOWN_20_TRADING_DAYS
```

## 10. 实盘晋级门槛

v1.2 只有在下列条件全部满足后，才能单独审查是否予以小额实盘权限：

1. 完成不少于60个交易日的前向纸面验证；
2. 完成交易不少于40笔，并有不少于20个近似独立样本；
3. 计入佣金、最低佣金、印花税、滑点、涨跌停锁定和未成交；
4. 滚动样本外结果的期望净收益为正，且不依赖单一参数点或少数极端交易；
5. `10:30` 或 `11:00` 扩展组在新样本上优于原 `10:00` 基准组；
6. 数据状态为 `STRICT`，不得使用 `PROXY` 结果晋级。

## 11. 快照状态机

```text
BUILD_UNIVERSE
→ SET_AS_OF_TO_LAST_COMPLETE_MINUTE
→ BUILD_SAME_ELAPSED_MINUTE_BASELINE
→ CALCULATE_DYNAMIC_C6
→ REQUIRE_TWO_CONSECUTIVE_MINUTES
→ FRESH_SIGNAL_3_MINUTES
→ WAIT_CURRENT_PULLBACK
→ ENTRY_OR_MISSED_ENTRY
→ NO_CHASE
→ OVERNIGHT
→ T_PLUS_1_EXIT
```
