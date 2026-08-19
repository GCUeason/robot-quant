# C2-A 云端部署

## 结论

C2-A 的准点调度必须运行在持续在线、具有持久磁盘的 Linux 主机上。GitHub 保存代码、
轻量 JSON/CSV/Markdown 报告和审计历史，但不保存 BigQuant 凭证、SSH 私钥、原始分钟数据、
`data/c2a/`、`data/c2a_fast/` 或 `parameter_paths/`。

GitHub Actions 的 `schedule` 可能延迟或丢失，本项目不把它作为 10:03 主时钟。仓库中的
systemd timers 使用 `Asia/Shanghai` 固定执行：

| 阶段 | 时间 | 入口 | 主要产物 |
|---|---|---|---|
| 盘前准备 | 工作日 08:45 | `c2a_cloud prepare` | `c2a_prepare_latest.md` |
| 模拟扫描 | 工作日 10:02:30 启动，目标 10:03 前完成 | `c2a_cloud scan` | `fast_latest.json`、日期化盘中报告 |
| 盘后复盘 | 工作日 16:30 | `c2a_cloud review` | 当日收盘对账 |
| 盘后研究 | 工作日 16:35 | `c2a_cloud research` | 研究报告和下一日基线状态 |

所有阶段固定为 `PAPER_ONLY`。错误时点、基线过期、覆盖不足或行情缺失都会写入当日
`DATA_NOT_READY` 空候选，禁止沿用上一日信号。16:30/16:35 阶段还要求行情日期为当日且
时间不早于 15:00；周一至周五 timer 落在法定休市日时会失败关闭，不会把休市日当交易日。

## 云端主机要求

- 受支持的 Linux 发行版和 systemd；
- 持续在线，不使用会自动休眠的交互式 Notebook/AIStudio 容器作为主时钟；
- Python 3.11 或更高版本、Git、OpenSSH、rsync、flock、GNU `timeout`；
- `/srv/robot-quant` 为 GitHub 仓库克隆目录，`.venv` 已安装 `.[dev]`；
- Git `origin` 使用不含内嵌 Token 的 SSH URL（`git@github.com:GCUeason/robot-quant.git`）；
- `robotquant` 服务用户拥有仓库目录；
- 两把用途隔离的密钥：
  - 只连接 BigQuant AIStudio 的 SSH 私钥；
  - 只对 `GCUeason/robot-quant` 有写权限的 GitHub deploy key；
- BigQuant 主机公钥固定写入 `known_hosts`，不得在每次运行时临时信任 `ssh-keyscan` 结果。

私钥、Token、AK/SK 和环境文件权限必须为 `0600`，不得写入仓库、日志、Actions cache
或 artifact。

## 环境配置

`/etc/robot-quant/c2a.env` 示例（不包含密钥）：

```bash
ROBOT_QUANT_PROJECT_ROOT=/srv/robot-quant
ROBOT_QUANT_PYTHON=/srv/robot-quant/.venv/bin/python
C2A_SSH_HOST=bigquant-aistudio
C2A_REMOTE_ROOT=/home/aiuser/work/robot-quant
```

SSH 别名由云端服务用户自己的 `~/.ssh/config` 提供。私钥路径和实际用户名仅保存在云端，
不进入此文件示例或 GitHub。

## 安装 timers

```bash
sudo install -o root -g root -m 0644 \
  deploy/systemd/robot-quant-c2a@.service \
  deploy/systemd/robot-quant-c2a-prepare.timer \
  deploy/systemd/robot-quant-c2a-scan.timer \
  deploy/systemd/robot-quant-c2a-review.timer \
  deploy/systemd/robot-quant-c2a-research.timer \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now \
  robot-quant-c2a-prepare.timer \
  robot-quant-c2a-scan.timer \
  robot-quant-c2a-review.timer \
  robot-quant-c2a-research.timer
```

核验下一次执行时间：

```bash
systemctl list-timers 'robot-quant-c2a-*'
```

先分别手工验证服务，再等待一次真实交易日自动运行：

```bash
sudo systemctl start robot-quant-c2a@prepare.service
sudo systemctl start robot-quant-c2a@scan.service
sudo systemctl start robot-quant-c2a@review.service
sudo systemctl start robot-quant-c2a@research.service
journalctl -u 'robot-quant-c2a@*' --since today
```

非对应时段的手工验证应使用 Python 入口的 `--allow-off-schedule`，且产物会继续标记为
模拟研究；systemd 服务本身不使用该开关。

## 写回与并发

`scripts/run_c2a_cloud_phase.sh` 使用单机 `flock` 防止四个阶段重叠，只暂存明确列出的
C2-A 结果，提交后推送 `main`。前置同步、虚拟环境或阶段进程失败时，也会先写当日
`DATA_NOT_READY`；若项目虚拟环境损坏，则由只使用标准库的失败记录器兜底。服务失败后
systemd 每两分钟重试，30 分钟内最多三次。锁只等待 90 秒；锁冲突或仓库脏树时，通过
临时干净 clone 写出失败状态，避免旧 `latest` 或用户暂存文件混入提交。若现有 GitHub
日报同时更新，云端脚本和 Actions 都执行最多三次 `fetch + rebase + push`；冲突会失败
并保留证据，不自动覆盖远端。

每个 JSON 状态包含 `scheduled_at`、`started_at`、`finished_at`、`source_commit` 和
`payload_sha256`，并在快速基线存在时记录 `source_manifest_sha256`。`retryable=true` 的
`DATA_NOT_READY/PARTIAL` 会让服务返回临时失败码，
从而触发上述有限重试；错误时点等确定性失败不会无限重跑。

现有 `.github/workflows/daily.yml` 同样使用结果白名单。不要恢复 `git add data/ reports/`
这类宽泛写法。

## 上线验收

完成部署不能只看 timer 为 active，必须在一个真实交易日逐项验证：

1. 08:45 产物日期正确，`baseline_as_of` 为上一交易日；
2. 10:02:30 启动并目标在 10:03 前生成当日日期化扫描；超时标记 `LATE`，失败时 `entries=[]` 且没有旧信号；
3. 16:30 读取同一天早盘扫描，并且只用当日 15:00 后快照复盘；若来源为 `LATE / RECONSTRUCTED`，结果必须标记 `RECONSTRUCTED_REVIEW`，不得算作准点闭环；
4. 16:35 独立启动研究流水线，下一交易日基线截止更新到当日；
5. 四阶段报告均由云端提交到 GitHub，可从另一台电脑查看；
6. 在一个工作日休市场景中确认盘后阶段为 `DATA_NOT_READY`；
7. 仓库历史中不存在私钥、Token、原始分钟数据和 pickle 缓存；
8. 模型门槛仍为 `FAIL / PAPER_ONLY`，没有券商连接。
