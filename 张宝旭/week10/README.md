# 本地知识助手（RAG + Chat）

一个本地小服务，浏览器里两个页面：
- **💬 问答**：你提问 → 在当前知识库里检索 → 命中片段 + 问题一起喂给大模型 → 基于知识库的回答（不会乱编）
- **📚 知识库**：富文本编辑（粘贴/拖拽图片）、上传 Word 整库覆盖、导出当前知识库为 Word

支持**多个独立知识库**（默认带 `WAF 知识库` 和 `大模型 FAQ` 两个），顶栏可切换、可新建、可删除，每个库的数据和系统提示词都互相隔离。

## 启动

- Windows：双击 `start.bat`
- macOS：双击 `start.command`

服务地址：**http://127.0.0.1:7321/**（端口在 `config.json` 里改）。

第一次跑会自动 `pip install python-docx`，之后秒开。

## 配置

编辑 `config.json`：

```json
{
  "host": "127.0.0.1",
  "port": 7321,
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "sk-xxxx",
  "model": "deepseek-v4-flash",
  "temperature": 0.3,
  "top_k": 5,
  "verify_ssl": false,
  "system_prompt": "..."
}
```

- `port`：被占用就改个数
- `base_url` 末尾别带 `/chat/completions`
- `verify_ssl: false` 用于公司有 HTTPS 中间人代理时绕过证书校验
- `system_prompt`：所有库的兜底；每个库自己的提示词在 `kb/index.json` 里改
- 改完 `model/api_key/system_prompt` 不用重启；改 `port/host` 要重启

## 知识库怎么维护

| 方式 | 怎么做 | 效果 |
|---|---|---|
| 上传 Word | 知识库页 → "⇡ 导入 Word"，输入"覆盖"二字确认 | **整库覆盖**：清空当前库，按新 Word 重建 |
| 网页里编辑 | 点条目 → "✎ 编辑" | 富文本（粗体/列表/标题/链接），图片可粘贴/拖拽/点按钮 |
| 网页里新增 | "＋ 新增条目" | 同上 |

**导出 Word**：点"⇣ 导出 Word"，把当前库打包成 docx，下次还能上传回来 round-trip。

> 建议：上传新 Word 前先点一次"导出 Word"做备份。

## 多知识库

- 顶栏右上的下拉切换库
- 选 `+ 新建知识库...` 只填名字（id 自动生成），就有了一个空库
- 点 🗑 可以删当前库（要输入库名二次确认；最后 1 个不让删）
- 库注册表是 `kb/index.json`，每个库的 `system_prompt` 可以单独改

## 打包发给别人

双击 `package.bat`，会问三种打法：
1. 含数据 + 含 api_key：自己备份用
2. 含数据 + 不含 api_key：发给同事，他们填自己的 key
3. 干净壳 + 不含 api_key：让对方上传自己的 Word 从零开始

zip 在 `dist/`。同事拿到：
- Windows：解压 → 双击 `start.bat`
- Mac：解压 → 双击 `start.command`（系统拦截就 Control 点一次）
- 任一平台：第一次运行自动 `pip install python-docx`

## 目录

```
wyquestion/
├─ config.json                    # 模型 / API key / 端口
├─ server.py                      # 本地服务（Python 标准库 + python-docx）
├─ start.bat / start.command      # Windows / Mac 启动脚本
├─ package.bat                    # 打包给同事
├─ tools/
│  ├─ build_data.py               # docx → entries.json
│  ├─ export_docx.py              # entries.json → docx
│  └─ package.py                  # 打包脚本
├─ kb/
│  ├─ index.json                  # 库注册表（库 id / 显示名 / system_prompt）
│  ├─ waf/                        # 一个库的目录
│  │   ├─ entries.json
│  │   └─ images/
│  └─ llm/                        # 另一个库的目录
└─ web/
   ├─ index.html  style.css  app.js
   └─ vendor/quill.js  quill.snow.css   # 富文本编辑器
```

## 安全

- API key 只在本机的 `config.json`，不要 commit 到 git（git 已经设置 `skip-worktree`）
- 服务只监听 `127.0.0.1:7321`，外网/同事电脑访问不到
- 富文本落库前会做白名单清洗，挡 `<script>` / `onerror=` / `javascript:` 等

## HTTP 接口（脚本调用用）

下面所有路径都接受 `?kb=<kbid>` 参数，省略时取默认库。

- `GET  /api/kbs` 列出所有知识库
- `POST /api/kbs` 新建（body: `{id, name, system_prompt?}`）
- `PUT  /api/kbs?kb=...` 改名/改 prompt（body: `{name?, system_prompt?}`）
- `DELETE /api/kbs?kb=...` 删库（连数据一起）
- `GET  /api/entries` 条目列表
- `GET  /api/entry?anchor=` 详情
- `POST /api/entry` 新增/更新（body: `{anchor?, title, html?, content?, tags}`）
- `DELETE /api/entry?anchor=`
- `POST /api/upload` 图片（multipart, field=`file`）→ 返回 `{url}`
- `POST /api/import-docx` 上传 Word 整库覆盖（multipart, field=`file`）
- `GET  /api/export-docx` 下载当前库为 docx
- `POST /api/search` 检索（body: `{query, top_k, kb}`）
- `POST /api/chat` RAG + LLM（body: `{query, history?, top_k?, kb}`）
