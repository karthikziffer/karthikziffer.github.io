---
layout: post
title: "Logging claude code file access with MCP"
author: "Karthik"
categories: journal
tags: [agentic ai, prompt engineering]
---

I recently read [this post](https://blog.pragmaticengineer.com/grolk-cli-uploaded-all-your-files-to-the-cloud/) about a CLI agent that uploaded local files to the cloud, without anyone asking it to. That's one of the biggest risks with coding agents without traceability and auditing. The agent decides what to read, and the user just trusts it.

So I built a small fix for my local Claude Code setup. I decide to control claude code to read the files and log which files the agent can read and write, and every access gets logged. This post walks through how it works.

---

## How Claude Code reads files today

Claude Code reads files two ways: through the `Read` tool, which is built directly into the CLI and is the faster path, or through Bash, which spins up a child process to run `cat`.

**The `Read` tool.**

1. Claude Code calls `open()` then `read()` directly, in its own process.
2. There's no separate process, no protocol, no messages passed anywhere, it's just the CLI touching disk, the same way any normal program would.
3. The file's text is returned straight to the model.

**`cat` via Bash.**

1. Claude Code opens a shell (`/bin/bash` or similar) as a child process.
2. The shell runs `cat local_file.txt` as its own child process, the real `cat` binary doing its own `open()` and `read()`.
3. `cat` prints the file to stdout.
4. The shell captures that output.
5. Claude Code's Bash tool captures the shell's output and passes it back to the model.

Neither path is logged or blocked by default. The fix is to block both and route every read through an MCP server tool instead.

<figure class="diagram">
<svg viewBox="0 0 700 330" role="img" aria-label="Diagram comparing file access without the MCP server, which has two unlogged paths, and with the MCP server, which has one logged path" class="diagram">
  <defs>
    <marker id="arrowHead1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <text x="0" y="16" class="lane-title">without the mcp server</text>

  <rect x="0" y="30" width="140" height="36" rx="3" class="box"></rect>
  <text x="70" y="52" text-anchor="middle" class="label">Claude Code</text>

  <rect x="280" y="6" width="150" height="36" rx="3" class="box"></rect>
  <text x="355" y="28" text-anchor="middle" class="label">Read tool</text>

  <rect x="280" y="62" width="150" height="36" rx="3" class="box"></rect>
  <text x="355" y="84" text-anchor="middle" class="label">Bash → cat</text>

  <rect x="540" y="30" width="150" height="36" rx="3" class="box"></rect>
  <text x="615" y="52" text-anchor="middle" class="label">Local files</text>

  <path d="M140,42 L280,24" class="arrow" marker-end="url(#arrowHead1)"></path>
  <path d="M140,54 L280,80" class="arrow" marker-end="url(#arrowHead1)"></path>
  <path d="M430,24 L540,42" class="arrow" marker-end="url(#arrowHead1)"></path>
  <path d="M430,80 L540,54" class="arrow" marker-end="url(#arrowHead1)"></path>

  <text x="615" y="86" text-anchor="middle" class="label-muted">not logged</text>

  <line x1="0" y1="150" x2="700" y2="150" class="lane-rule"></line>

  <text x="0" y="176" class="lane-title">with the mcp server</text>

  <rect x="0" y="190" width="140" height="36" rx="3" class="box"></rect>
  <text x="70" y="212" text-anchor="middle" class="label">Claude Code</text>

  <rect x="280" y="184" width="150" height="48" rx="3" class="box-accent"></rect>
  <text x="355" y="204" text-anchor="middle" class="label-strong">MCP server</text>
  <text x="355" y="221" text-anchor="middle" class="label-small">read_file / write_file</text>

  <rect x="540" y="190" width="150" height="36" rx="3" class="box"></rect>
  <text x="615" y="212" text-anchor="middle" class="label">Local files</text>

  <rect x="280" y="264" width="150" height="36" rx="3" class="box"></rect>
  <text x="355" y="286" text-anchor="middle" class="label">access log</text>

  <path d="M140,208 L280,208" class="arrow" marker-end="url(#arrowHead1)"></path>
  <path d="M430,208 L540,208" class="arrow" marker-end="url(#arrowHead1)"></path>
  <path d="M355,232 L355,264" class="arrow" marker-end="url(#arrowHead1)"></path>

  <text x="355" y="320" text-anchor="middle" class="label-accent">every call logged, in or out</text>
</svg>
<figcaption>Without the server, reads happen through two unlogged paths. With it, every read and write funnels through one server that logs the call.</figcaption>
</figure>

---

## Blocking the default read paths

This has to apply in a global scope, so both files go in the home directory. There are also configuration to handle it for project and local scope. It depends on where the settings and CLAUDE.md file is placed and claude code accesses it. 

**`~/.claude/settings.json`** blocks the two default read paths.

```json
"permissions": {
  "deny": ["Read", "Bash"]
}
```

**`~/CLAUDE.md`** tells Claude Code what to use instead.

```
When you need to read a file's contents, use the `read_file` tool
from the `claude-file-reader` MCP server instead of any built-in file-reading tool.
```

With both in place, Claude Code has no default way left to read a file. It has to go through the MCP tool.

This same `deny` rule also blocks Claude Code's built-in `Write` and `Edit` tools, since both check a file with a read before touching it. So once the built-in read path is gone, the built-in write path is gone with it, which is why the server needs a write tool of its own.

---

## The MCP server

The MCP server handles the read and write tool and also log it in parallel. This provides the user with logging of the files accessed at every call. 


Here's how the server is set up.

```python
from datetime import datetime
from pathlib import Path
from mcp.server import MCPServer
import json
import logging
import uuid

mcp = MCPServer("claude-file-reader")
```

The allowed root is the only folder Claude Code can read from. Anything outside it gets blocked.

```python
ALLOWED_ROOT = Path.home().resolve()
```

Every read gets logged with metadata for auditing.

```python
LOG_PATH = Path.home() / ".claude" / "claude-file-reader-access.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
SESSION_ID = uuid.uuid4().hex[:12]

logger = logging.getLogger("claude-file-reader")
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

def log_access(*, tool: str, 
    document: str, 
    operation: str, 
    allowed: bool, 
    bytes_returned: int = 0) -> None:
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": SESSION_ID,
        "tool": tool,
        "document": document,
        "operation": operation,
        "bytes_returned": bytes_returned,
        "allowed": allowed,
    }
    logger.info(json.dumps(entry))
```

A counter tracks how many reads and writes have happened in the session.

```python
_call_count = 0
```

`@mcp.tool()` turns this function into an MCP tool. It resolves the path, rejects anything outside `ALLOWED_ROOT`, reads the file, logs it, and returns the content.

```python
@mcp.tool()
def read_file(rel_path: str) -> str:
    """..."""
    global _call_count
    _call_count += 1
    target = (ALLOWED_ROOT / rel_path).resolve()
    if not target.is_relative_to(ALLOWED_ROOT):
        raise ValueError(f"{rel_path} is outside the allowed root")
    if not target.is_file():
        raise FileNotFoundError(f"No such file: {rel_path}")
    content = target.read_text()
    log_access(tool="read_file", document=rel_path, operation="read",
                allowed=True, bytes_returned=len(content.encode("utf-8")))
    return f"[call #{_call_count}] {content}"
```

---

## Adding a write tool

`write_file` follows the same pattern as `read_file`. Same root check, same log line, just a write instead of a read.

```python
@mcp.tool()
def write_file(rel_path: str, content: str) -> str:
    """..."""
    global _call_count
    _call_count += 1
    target = (ALLOWED_ROOT / rel_path).resolve()
    if not target.is_relative_to(ALLOWED_ROOT):
        raise ValueError(f"{rel_path} is outside the allowed root")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    log_access(tool="write_file", document=rel_path, operation="write",
                allowed=True, bytes_returned=len(content.encode("utf-8")))
    return f"[call #{_call_count}] wrote {len(content.encode('utf-8'))} bytes to {rel_path}"
```

It blocks the same way `read_file` does, anything outside `ALLOWED_ROOT` gets blocked, and it logs the same way too, just with `operation="write"` instead of `"read"`. 

1. Add `write_file` to `server.py`.
2. Reconnect the server: `/mcp` → `claude-file-reader` → `2. Reconnect`.
3. The new tool shows up alongside `read_file`. No change to `settings.json` or `CLAUDE.md` is needed.


---

## Registering the server


The mcp server is registered to the Claude code with the following command. When the Claude code session starts, it spins up a subprocess to run the `server.py`. Then it creates an handshake and lists the tools, read_file and write_file with the parameters to call each tools. 

```bash
claude mcp add claude-file-reader --scope user -- uv --directory ../claude-file-reader run server.py
```

`--scope user` is the part that matters. It writes the server into Claude Code's own config, outside any single project, so this applies everywhere you use Claude Code, not just in one repo.

---

## Checking the connection

Run `/mcp` inside Claude Code to see the list of connected mcp servers. 

```
Manage MCP servers
  5 servers

    User MCPs (/Users/karthik/.claude.json)
  ❯ claude-file-reader · ✔ connected · 2 tools
```

Selecting it shows the connection details.

```
Claude-file-reader MCP Server

  Status:  ✔ connected
  Command: uv
  Args:  --directory ../Projects/claude-file-reader run server.py
  Config location:  /Users/karthik/.claude.json
  Capabilities: tools
  Tools: 2 tools

  ❯ 1. View tools
    2. Reconnect
    3. Disable
```

From here you can reconnect or disable it like any other MCP server. From now on, every file Claude Code reads or writes goes through `read_file` or `write_file`, and gets logged.

---

## What the logs look like

Each entry is a single JSON line, one per read or write.

```json
{"timestamp": "2026-08-25T19:50:51+02:00", "session_id": "18ab7cd85edb", "tool": "read_file", "document": "/Users/karthik/Projects/2026-08-25-logging-claude-code-file-access-with-mcp.md", "operation": "read", "bytes_returned": 148, "allowed": true}


{"timestamp": "2026-08-25T19:52:03+02:00", "session_id": "18ab7cd85edb", "tool": "write_file", "document": "/Users/karthik/Projects/2026-08-25-logging-claude-code-file-access-with-mcp.md", "operation": "write", "bytes_returned": 6431, "allowed": true}
```

With this, every read and write Claude Code makes is auditable afterward. You can see which file, when, how much data, whether it was a read or a write, and whether it was allowed.

---

## The tradeoff

The MCP server is its own process. Claude Code launches `uv --directory ... run server.py` as a subprocess. Everything after that runs over a real protocol, starting with a handshake, then JSON-RPC messages over stdin/stdout for the `list_tools`/`call_tool` exchange.

<figure class="diagram">
<svg viewBox="0 0 700 190" role="img" aria-label="Diagram showing Claude Code and the MCP server communicating as separate processes over JSON-RPC on stdin and stdout" class="diagram">
  <defs>
    <marker id="arrowHead2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="10" y="70" width="160" height="44" rx="3" class="box"></rect>
  <text x="90" y="97" text-anchor="middle" class="label">Claude Code</text>

  <rect x="290" y="20" width="380" height="140" rx="3" class="box-dashed"></rect>
  <text x="306" y="40" class="lane-title">separate process</text>

  <rect x="330" y="72" width="300" height="44" rx="3" class="box-accent"></rect>
  <text x="480" y="99" text-anchor="middle" class="label-strong">uv run server.py</text>

  <text x="480" y="135" text-anchor="middle" class="label-small">list_tools · call_tool</text>

  <path d="M170,90 L290,90" class="arrow" marker-end="url(#arrowHead2)" marker-start="url(#arrowHead2)"></path>
  <text x="230" y="76" text-anchor="middle" class="label-muted">JSON-RPC</text>
  <text x="230" y="118" text-anchor="middle" class="label-muted">stdin / stdout</text>
</svg>
<figcaption>Claude Code and the MCP server talk over a real protocol, a handshake followed by JSON-RPC calls, since the server runs as its own subprocess.</figcaption>
</figure>

The MCP server introduces a gateway to log every read and write.
