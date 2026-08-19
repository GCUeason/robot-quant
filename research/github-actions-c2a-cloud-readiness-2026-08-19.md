# C2-A 云端持续运行与 GitHub Actions 就绪性审计

审计时间：2026-08-19（Asia/Shanghai）
仓库：`GCUeason/robot-quant`
范围：只读核查 GitHub Actions、Secrets、Variables、Environments、缓存、Artifacts、默认分支与 SSH 依赖；未修改任何远端状态。

## 结论

当前部署状态是 **NO-GO**：GitHub 上只有 ETF/产业链日更工作流，C2-A 代码仍只在本机未跟踪文件中；仓库级 Actions Secrets、Variables、Environments 和 Deploy Keys 均为空；不存在 C2-A 持久状态或报告 Artifact。

目标架构本身 **可行，但不应把 GitHub 托管 `schedule` 当作 10:03/16:35 的主时钟**。推荐使用一台持续在线、带持久磁盘的 Linux 云主机，由 `systemd` timer 在 Asia/Shanghai 时区运行盘前准备、10:03 纸面扫描和 16:30/16:35 盘后更新/复盘；GitHub 用于代码、CI、轻量结果与审计报告。GitHub Actions 可保留为手动补跑和非精确时点任务。

## 当前 GitHub 实况

| 项目 | 2026-08-19 只读结果 | 影响 |
|---|---:|---|
| 仓库可见性 | public | 标准 GitHub-hosted runner 免费且不限分钟，但代码和提交的结果均公开 |
| 默认分支 | `main` | 定时工作流只从默认分支最新提交运行 |
| 远端 `main` | `310b5b7` | 本机仍在 `1543e79`，整理 C2-A 前必须先安全整合远端机器人提交 |
| 分支保护 | 无 | 在接入 SSH 私钥前应先收紧主分支与工作流修改权限 |
| Actions | enabled，`allowed_actions=all` | 当前允许任意公开 Action |
| 强制 SHA pin | false | 当前 `checkout@v6`、`setup-python@v6` 不是不可变引用 |
| 默认 `GITHUB_TOKEN` | read | `daily.yml` 显式声明 `contents: write`，已有机器人推送成功 |
| 工作流 | 仅 `Daily prediction and simulation` | cron 为 `30 8 * * 1-5`，没有 C2-A 分支 |
| Actions Secrets | 0 | GitHub runner 无法登录 BigQuant AIStudio |
| Actions Variables | 0 | SSH host/user/remote root/时区均未配置 |
| Environments | 0 | 没有 C2-A 环境级权限或密钥隔离 |
| Deploy Keys | 0 | 持续在线云主机目前没有专用仓库写回凭证 |
| Actions cache | 1 个，约 127.6 MB | 只有 pip 缓存，没有 C2-A 滚动状态缓存 |
| Artifacts | 0 | 没有 C2-A 运行快照或失败证据 |
| Artifact/log retention | 90 天 | 公共仓库允许的当前最大值也是 90 天 |

现有 `daily.yml` 会校验、运行 `run-daily`，然后广泛执行 `git add data/ reports/` 并推送 `main`。C2-A 上云后不能继续依赖这个宽泛暂存范围，否则可能把不应公开的行情分区、缓存或诊断文件一并提交；应改为严格的结果白名单。

### 已观察到的定时延迟

当前工作流标称 08:30 UTC（北京时间 16:30）。最近 20 次 `schedule` 运行的实际启动延迟：

- 最小：31.6 分钟
- 中位数：126.8 分钟
- 最大：203.8 分钟
- 平均：113.7 分钟

最近 9 次成功运行也分别延迟约 31、36、63、68、67、58、85、59、148 分钟。该仓库的真实运行记录已经足以否定“GitHub cron 会在指定分钟准时启动”的假设。

GitHub 官方也明确说明：`schedule` 在高负载时会延迟，严重时排队任务可能被丢弃；它只在默认分支运行，公共仓库 60 天无活动会自动停用。GitHub 现在支持 IANA `timezone`，并允许最短 5 分钟频率，但这些能力并不提供准点 SLA。[触发事件说明](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)；[工作流语法](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule)

## 持久化边界

GitHub-hosted runner 每个 job 都使用新 VM，job 结束后 VM 被销毁，因此本地工作目录不能作为下一次扫描的状态。[GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)

当前本机快速包约 24 MB，包含：

- `data/c2a_fast/rolling_state.npz`
- `data/c2a_fast/universe.csv.gz`
- `data/c2a_fast/manifest.json`

推荐边界：

1. **权威滚动状态**：放在云主机持久磁盘；原子写入并保留上一版可恢复快照。
2. **BigQuant 研究缓存**：继续留在 AIStudio 工作空间；云主机只通过 SSH 编排，不把全年分钟原始分区提交到 GitHub。
3. **GitHub 仓库**：只提交轻量、可公开、可审计的 C2-A JSON/CSV/Markdown 结果，且显式白名单暂存。
4. **Actions cache**：只能缓存可重新生成的数据。官方要求调用方在 cache 缺失时仍可重建；默认 7 天未访问会删除，仓库默认总上限 10 GB；缓存内容应视为不可信，绝不能包含密钥。[缓存用途与安全](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching)；[缓存保留与淘汰](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#usage-limits-and-eviction-policy)
5. **Artifacts**：用于保存每次运行的日志、状态快照或失败诊断，不作为无限期连续账本。当前公共仓库上限 90 天；跨 workflow/run 下载还需要 token 与 run id。[Artifacts 用途](https://docs.github.com/en/actions/tutorials/store-and-share-data)；[仓库保留期](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#configuring-the-retention-period-for-github-actions-artifacts-and-logs-in-your-repository)

## SSH 与凭证方案

本机已配置专用 BigQuant ED25519 身份文件，权限为 `0600`；SSH alias 开启 `IdentitiesOnly` 和严格 host-key 校验，`known_hosts` 中已有目标主机条目。不要把该私钥或 `known_hosts` 内容提交到仓库、cache 或 artifact。

### 推荐：云主机直连

1. 为云主机生成一把独立的 ED25519 密钥，避免复制 Mac 私钥；在 BigQuant 侧只授权这把云端公钥，便于单独撤销。
2. 私钥保存在云主机专用服务账号目录，权限 `0600`；固定 `known_hosts` 指纹并启用 `StrictHostKeyChecking=yes`、`BatchMode=yes`。
3. 云主机拉取公开代码不需要凭证；写回报告优先使用权限最小、短期的 GitHub App token。简单方案可用仅限此仓库的 write deploy key，但 GitHub 官方指出 deploy key 默认只读、可选写权限、不会过期，服务器失陷时风险更高。[Deploy keys 与 GitHub App](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)

### 备选：GitHub Actions 直连 BigQuant

如果仅用于手动补跑，可配置：

- Secrets：`BIGQUANT_SSH_PRIVATE_KEY`，如密钥有口令再增加单独的 passphrase secret。
- Variables：`BIGQUANT_SSH_HOST`、`BIGQUANT_SSH_USER`、`BIGQUANT_REMOTE_ROOT`、`C2A_TIMEZONE=Asia/Shanghai`。
- Environment：`c2a-cloud`，只允许默认分支使用；全自动任务不能设置每次人工批准。
- 独立、预先核验的 host key：从可信渠道写入 `known_hosts`，不要在同一不可信连接上临时 `ssh-keyscan` 后直接信任。

Secret 应通过环境变量或 STDIN 写入权限受限的临时文件，不能放在命令行参数。GitHub 明确说明 secret 只有在 workflow 显式引用时才注入，缺失 secret 会解析为空字符串，而且日志脱敏并非所有变换场景都绝对可靠。[在 Actions 中使用 Secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets)；[Secrets 安全模型](https://docs.github.com/en/actions/concepts/security/secrets)

在任何 SSH secret 接入前，至少先完成：

- 保护 `main` 或建立等价 ruleset，避免未经审查的 workflow 修改进入默认分支。
- 将 Actions 限制为 GitHub-owned actions；将 `checkout`、`setup-python`、`upload-artifact` 等 pin 到完整 commit SHA。GitHub 官方称完整 SHA 是使用 Action 不可变发布的唯一方式。[Secure use reference](https://docs.github.com/en/actions/reference/security/secure-use#using-third-party-actions)
- 工作流 job 只授予需要的 `contents` 权限；测试 job 为 read，单独的报告写回 job 才为 write。

## 推荐运行拓扑

```text
持续在线 Linux VM（systemd timers，Asia/Shanghai，持久磁盘）
  盘前准备 -> 校验上一交易日基线与 manifest
       |
       +-> 10:03 C2-A PROXY / PAPER_ONLY 扫描
       |      -> 同日 JSON/Markdown 报告
       |
       +-> 16:30 BigQuant 增量更新
              -> 16:35 或更新完成后复盘
                     -> 轻量结果白名单写回 GitHub

GitHub Actions
  push/PR: Ruff + pytest + 安全检查
  workflow_dispatch: 手动补跑/诊断
  非精确 schedule: ETF/产业链日更或兜底巡检
```

## 上线必须满足的门槛

1. **代码门槛**：C2-A 源码、测试、CLI 与 README 安全纳入版本控制；不包含私钥、token、AIStudio 配置、原始分钟数据或本机 IDE 私有文件。
2. **时间门槛**：云主机 NTP 正常，timer 使用 `Asia/Shanghai`；工作日只是粗过滤，程序必须用有效 A 股交易日校验。错时或迟到超过允许窗口时输出 `NO-GO / DATA_NOT_READY`，不得补用昨天候选。
3. **数据门槛**：盘前 `manifest.last_processed_date` 必须等于上一有效交易日；覆盖率、完整分钟和同分钟 20 日基线继续 fail closed。
4. **模拟边界**：所有输出固定 `PAPER_ONLY`、`real_trade_authorized=false`；GitHub/SSH/调度成功不等于模型门槛通过。
5. **并发门槛**：VM 端使用单实例锁；16:30 现有 GitHub workflow 与 C2-A 写回不能并发覆盖同一文件。写回前 fetch/rebase，并对非快进冲突失败退出。
6. **公开边界**：只提交明确白名单结果，例如 `reports/c2a_*.md` 与 `data/c2a_results/*latest*.json`；不再对 C2-A 使用 `git add data/ reports/`。
7. **可观测性**：每次报告记录 `scheduled_at`、`started_at`、`finished_at`、交易日、数据截止、状态、失败原因、源 manifest hash 和代码 SHA。即使无候选或数据未就绪，也要写出同日状态报告。
8. **恢复门槛**：云主机重启后 timer 自动恢复；持久状态损坏或丢失时必须回退到上一快照或重新准备，不能生成候选。
9. **验收门槛**：先完成干跑和手动触发，再以至少一个真实交易日的四段证据验收：盘前准备完成、10:03 同日扫描、16:30/16:35 同日复盘、GitHub 上可见且可追溯的报告。

## 只读复核命令

以下调用用于本次审计，均为 GET/本地读取：

```bash
gh api "repos/GCUeason/robot-quant"
gh api "repos/GCUeason/robot-quant/branches/main"
gh api "repos/GCUeason/robot-quant/actions/workflows"
gh api "repos/GCUeason/robot-quant/actions/runs?event=schedule&per_page=20"
gh api "repos/GCUeason/robot-quant/actions/permissions"
gh api "repos/GCUeason/robot-quant/actions/permissions/workflow"
gh api "repos/GCUeason/robot-quant/actions/permissions/artifact-and-log-retention"
gh api "repos/GCUeason/robot-quant/actions/secrets"
gh api "repos/GCUeason/robot-quant/actions/variables"
gh api "repos/GCUeason/robot-quant/environments"
gh api "repos/GCUeason/robot-quant/actions/caches?per_page=100"
gh api "repos/GCUeason/robot-quant/actions/artifacts?per_page=100"
gh api "repos/GCUeason/robot-quant/keys"
```
