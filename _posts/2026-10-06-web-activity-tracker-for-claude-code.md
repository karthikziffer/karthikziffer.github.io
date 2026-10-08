---
layout: post
title: "Web activity tracker for Claude Code"
author: "Karthik"
categories: journal
tags: [claude-code, google-api]
---

When Claude Code searches the web, it works out of sight. Provides less visibility into the site's security, and the user can't tell whether those sites are safe. That matters, because whatever Claude reads goes straight into its context. A malicious page could carry a prompt injection: hidden text that tries to take over the session.

So I built a web activity tracker, a Claude Code mod (a small plugin that runs inside Claude Code). It does two things:

- **Shows Claude's web activity.** Every search Claude runs and every page it opens appears in a panel, along with the pages the answers were retrieved.
- **Screens each page.** Before Claude reads a page, the Claude Code mod checks its URL with the [Google Safe Browsing API](https://developers.google.com/safe-browsing), which compares URLs against Google's constantly updated lists of malware, phishing, and deceptive sites. Any page Google has flagged is blocked.

Safe Browsing won't catch a prompt injection hidden on an otherwise ordinary site. But it keeps Claude away from google flagged pages, and the panel lets me see and check every source Claude used.

In this post:

1. How Claude Code runs a web search
2. Where hooks fit in
3. How the Claude Code mod works
4. How to set up the Claude Code mod

---

## How Claude Code runs a web search

The user writes the search query, model decides the tools (web search), claude code calls the web search tool and retrieves context from the web pages and then the model generates the answer. 

### The basic flow

<figure class="diagram">
<svg viewBox="0 0 700 112" role="img" aria-label="Flow of a web search: the user asks, Claude picks a tool, Claude Code runs it, and Claude writes the answer" class="diagram">
  <defs>
    <marker id="ah-flow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="72.5" y="52" text-anchor="middle" class="label-strong">User</text>
  <text x="72.5" y="68" text-anchor="middle" class="label-small">asks a question</text>
  <rect x="185" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="257.5" y="52" text-anchor="middle" class="label-strong">Claude (model)</text>
  <text x="257.5" y="68" text-anchor="middle" class="label-small">picks WebSearch</text>
  <rect x="370" y="30" width="145" height="50" rx="3" class="box-accent"></rect>
  <text x="442.5" y="52" text-anchor="middle" class="label-strong">Claude Code</text>
  <text x="442.5" y="68" text-anchor="middle" class="label-small">runs the tool</text>
  <rect x="555" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="627.5" y="52" text-anchor="middle" class="label-strong">Claude (model)</text>
  <text x="627.5" y="68" text-anchor="middle" class="label-small">writes the answer</text>
  <path d="M145,55 L185,55" class="arrow" marker-end="url(#ah-flow)"></path>
  <path d="M330,55 L370,55" class="arrow" marker-end="url(#ah-flow)"></path>
  <path d="M515,55 L555,55" class="arrow" marker-end="url(#ah-flow)"></path>
  <text x="350" y="20" text-anchor="middle" class="label-small">tool call</text>
  <text x="535" y="20" text-anchor="middle" class="label-small">result</text>
  <text x="257.5" y="102" text-anchor="middle" class="label-muted">Anthropic&#x27;s servers</text>
  <text x="627.5" y="102" text-anchor="middle" class="label-muted">Anthropic&#x27;s servers</text>
</svg>
<figcaption>Claude decides which tool to call. Claude Code is what actually runs it.</figcaption>
</figure>

Every tool must have: 
- A name
- A description
- A JSON schema for its parameters. 

Claude Code sends these with every request. There are two types of tools available. 

### Normal tools and deferred tools

A session can have hundreds of tools, and sending every schema with every request adds up the context. So Claude Code splits tools into two kinds:

- **Normal tools:** Claude Code sends the full schema with every request.
- **Deferred tools:** Claude Code sends only the name, in a list at the start of the session.

The list of deferred tools looks like this:

```
The following deferred tools are now available via ToolSearch...
WebSearch
WebFetch
mcp__claude_ai_Gmail__send_message
mcp__claude-in-chrome__navigate
...
```

At this point, Claude knows each tool exists, but not what parameters it takes.

### How a deferred tool gets loaded

To use a deferred tool, Claude first calls `ToolSearch`, a tool that's always loaded:

```
ToolSearch(query="select:WebSearch")
```

`ToolSearch` returns the full schema. From then on, Claude can call the deferred tool like a normal tool. `ToolSearch` can also load several tools at once, or find tools by keyword.

<figure class="diagram">
<svg viewBox="0 0 700 100" role="img" aria-label="Loading a deferred tool: the session starts with tool names only, Claude calls ToolSearch, and the full schema makes WebSearch callable" class="diagram">
  <defs>
    <marker id="ah-deferred" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="30" width="170" height="50" rx="3" class="box"></rect>
  <text x="85" y="52" text-anchor="middle" class="label-strong">Session start</text>
  <text x="85" y="68" text-anchor="middle" class="label-small">tool names only</text>
  <rect x="265" y="30" width="170" height="50" rx="3" class="box-accent"></rect>
  <text x="350" y="52" text-anchor="middle" class="label-strong">ToolSearch</text>
  <text x="350" y="68" text-anchor="middle" class="label-small">select:WebSearch</text>
  <rect x="530" y="30" width="170" height="50" rx="3" class="box"></rect>
  <text x="615" y="52" text-anchor="middle" class="label-strong">WebSearch</text>
  <text x="615" y="68" text-anchor="middle" class="label-small">full schema, callable</text>
  <path d="M170,55 L265,55" class="arrow" marker-end="url(#ah-deferred)"></path>
  <path d="M435,55 L530,55" class="arrow" marker-end="url(#ah-deferred)"></path>
  <text x="217.5" y="20" text-anchor="middle" class="label-small">needs WebSearch</text>
  <text x="482.5" y="20" text-anchor="middle" class="label-small">returns schema</text>
</svg>
<figcaption>A deferred tool costs one extra round trip the first time Claude uses it.</figcaption>
</figure>

The trade-off: every request stays small, at the cost of one extra round trip the first time Claude uses a deferred tool.

### What happens when Claude calls WebSearch

When Claude decides to search, the model sends a tool call to Claude Code:

```json
{
  "type": "tool_use",
  "name": "WebSearch",
  "input": {
    "query": "What is the exact height of Mount Fuji?",
    "mode": "extended"
  }
}
```

Claude Code then runs the call through five steps, always in this order:

<figure class="diagram">
<svg viewBox="0 0 700 196" role="img" aria-label="The five steps of a tool call: schema check, PreToolUse hooks, permission check, tool runs, PostToolUse hooks. A failed schema check sends the error back to Claude." class="diagram">
  <defs>
    <marker id="ah-pipeline" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <text x="0" y="16" text-anchor="start" class="lane-title">every tool call, in order</text>
  <rect x="0" y="34" width="120" height="50" rx="3" class="box"></rect>
  <text x="60" y="63" text-anchor="middle" class="label">Schema check</text>
  <rect x="145" y="34" width="120" height="50" rx="3" class="box-accent"></rect>
  <text x="205" y="56" text-anchor="middle" class="label-strong">PreToolUse</text>
  <text x="205" y="72" text-anchor="middle" class="label-small">hooks</text>
  <rect x="290" y="34" width="120" height="50" rx="3" class="box"></rect>
  <text x="350" y="56" text-anchor="middle" class="label-strong">Permission</text>
  <text x="350" y="72" text-anchor="middle" class="label-small">check</text>
  <rect x="435" y="34" width="120" height="50" rx="3" class="box"></rect>
  <text x="495" y="63" text-anchor="middle" class="label">Tool runs</text>
  <rect x="580" y="34" width="120" height="50" rx="3" class="box"></rect>
  <text x="640" y="56" text-anchor="middle" class="label-strong">PostToolUse</text>
  <text x="640" y="72" text-anchor="middle" class="label-small">hooks</text>
  <path d="M120,59 L145,59" class="arrow" marker-end="url(#ah-pipeline)"></path>
  <path d="M265,59 L290,59" class="arrow" marker-end="url(#ah-pipeline)"></path>
  <path d="M410,59 L435,59" class="arrow" marker-end="url(#ah-pipeline)"></path>
  <path d="M555,59 L580,59" class="arrow" marker-end="url(#ah-pipeline)"></path>
  <path d="M60,84 L60,140" class="arrow" marker-end="url(#ah-pipeline)"></path>
  <text x="68" y="116" text-anchor="start" class="label-muted">fails</text>
  <rect x="0" y="140" width="120" height="50" rx="3" class="box"></rect>
  <text x="60" y="162" text-anchor="middle" class="label-strong">Error to Claude</text>
  <text x="60" y="178" text-anchor="middle" class="label-small">to fix the call</text>
  <text x="205" y="106" text-anchor="middle" class="label-accent">log, block, approve,</text>
  <text x="205" y="121" text-anchor="middle" class="label-accent">or rewrite</text>
</svg>
<figcaption>Every tool call goes through the same five steps. Hooks run before the permission check.</figcaption>
</figure>

The first step is the schema check. If the input doesn't match the tool's schema, Claude Code sends the input and the error back to Claude so it can fix the call. If the input matches, Claude Code moves on to the `PreToolUse` hooks.

---

## Hooks


A hook is a code that Claude Code runs each time a matching tool is about to run. A hook can:

- log the call
- block it
- decide whether it's allowed
- rewrite the input before the tool runs

Hooks are configured in `settings.json` or in a plugin. This one runs a logging script before every `WebSearch` call:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "WebSearch",
      "hooks": [{ "type": "command", "command": "~/.claude/hooks/log-search.sh" }]
    }]
  }
}
```

### Why hook order matters

Hooks run before the permission check. That means a hook can approve a call so the user is never prompted, or block it before a prompt appears. If a hook denies a call, no allow rule in the settings can override it.

### System hooks vs function hooks

The Claude Code mod uses function hooks. Claude Code has two kinds of hooks:

|                    | System hooks                              | Function hooks                                  |
|--------------------|-------------------------------------------|-------------------------------------------------|
| Configured in      | `settings.json`                           | A Claude Code mod                               |
| Written as         | A script, such as Python or Bash          | A JavaScript or TypeScript function             |
| How it runs        | As a separate program, fresh for each call | Inside Claude Code, staying loaded              |
| Keeps state        | No                                        | Yes                                             |
| Picks up code changes | On the next call                       | Hot-reloads                                     |

**System hooks** run before and after a tool. Each time, Claude Code starts the script, passes it some JSON, and reads its answer. The script then exits and keeps nothing.

**Function hooks** stay loaded between calls, so they can remember things. A function hook can wrap a tool call and keep state across calls. It can even skip the tool entirely and supply its own answer.

<figure class="diagram">
<svg viewBox="0 0 700 300" role="img" aria-label="System hooks run as a new process for each call. Function hooks are loaded once inside Claude Code and keep state." class="diagram">
  <defs>
    <marker id="ah-hooks" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <text x="0" y="16" text-anchor="start" class="lane-title">system hook</text>
  <rect x="0" y="34" width="140" height="56" rx="3" class="box"></rect>
  <text x="70" y="66" text-anchor="middle" class="label">Claude Code</text>
  <rect x="280" y="34" width="170" height="56" rx="3" class="box"></rect>
  <text x="365" y="59" text-anchor="middle" class="label-strong">Script</text>
  <text x="365" y="75" text-anchor="middle" class="label-small">starts, answers, exits</text>
  <path d="M140,52 L280,52" class="arrow" marker-end="url(#ah-hooks)"></path>
  <path d="M280,74 L140,74" class="arrow" marker-end="url(#ah-hooks)"></path>
  <text x="210" y="44" text-anchor="middle" class="label-small">JSON in</text>
  <text x="210" y="92" text-anchor="middle" class="label-small">answer</text>
  <text x="470" y="66" text-anchor="start" class="label-muted">new process for every call</text>
  <line x1="0" y1="140" x2="700" y2="140" class="lane-rule"></line>
  <text x="0" y="168" text-anchor="start" class="lane-title">function hook</text>
  <rect x="0" y="184" width="450" height="104" rx="3" class="box-dashed"></rect>
  <text x="16" y="206" text-anchor="start" class="label">Claude Code</text>
  <rect x="140" y="220" width="170" height="56" rx="3" class="box-accent"></rect>
  <text x="225" y="245" text-anchor="middle" class="label-strong">Mod function</text>
  <text x="225" y="261" text-anchor="middle" class="label-small">keeps state across calls</text>
  <text x="470" y="252" text-anchor="start" class="label-muted">loaded once, stays loaded</text>
</svg>
<figcaption>A system hook starts fresh for every call. A function hook stays loaded and remembers what it saw.</figcaption>
</figure>

### When to use which

- Use **system hooks** to share rules with a team, or to write the hook as a Python or Bash script.
- Use **function hooks** when you need memory between tool calls, or want to cache tool answers.

---

## How the Claude Code mod works

The Claude Code mod uses function hooks to track every search and page, and to check each page with the Google Safe Browsing API. It has three layers, and data flows in one direction:

<figure class="diagram">
<svg viewBox="0 0 700 210" role="img" aria-label="The Claude Code mod&#x27;s layers: Claude Code events trigger the hooks, the hooks check URLs with the Google Safe Browsing API and write stored state, and the panel reads the stored state" class="diagram">
  <defs>
    <marker id="ah-mod" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="72.5" y="52" text-anchor="middle" class="label-strong">Claude Code</text>
  <text x="72.5" y="68" text-anchor="middle" class="label-small">events</text>
  <rect x="185" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="257.5" y="52" text-anchor="middle" class="label-strong">Layer 1: Hooks</text>
  <text x="257.5" y="68" text-anchor="middle" class="label-small">record + check</text>
  <rect x="370" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="442.5" y="52" text-anchor="middle" class="label-strong">Layer 2: State</text>
  <text x="442.5" y="68" text-anchor="middle" class="label-small">session + $.store</text>
  <rect x="555" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="627.5" y="52" text-anchor="middle" class="label-strong">Layer 3: Panel</text>
  <text x="627.5" y="68" text-anchor="middle" class="label-small">green · blue · amber</text>
  <path d="M145,55 L185,55" class="arrow" marker-end="url(#ah-mod)"></path>
  <path d="M330,55 L370,55" class="arrow" marker-end="url(#ah-mod)"></path>
  <path d="M515,55 L555,55" class="arrow" marker-end="url(#ah-mod)"></path>
  <text x="165" y="20" text-anchor="middle" class="label-small">trigger</text>
  <text x="350" y="20" text-anchor="middle" class="label-small">write</text>
  <text x="535" y="20" text-anchor="middle" class="label-small">read</text>
  <rect x="162.5" y="150" width="190" height="50" rx="3" class="box-accent"></rect>
  <text x="257.5" y="172" text-anchor="middle" class="label-strong">Google Safe</text>
  <text x="257.5" y="188" text-anchor="middle" class="label-small">Browsing API</text>
  <path d="M245,80 L245,150" class="arrow" marker-end="url(#ah-mod)"></path>
  <path d="M270,150 L270,80" class="arrow" marker-end="url(#ah-mod)"></path>
  <text x="237" y="122" text-anchor="end" class="label-muted">URLs</text>
  <text x="278" y="122" text-anchor="start" class="label-muted">verdict</text>
</svg>
<figcaption>Data flows one way: hooks write the stored state, and the panel only reads it.</figcaption>
</figure>

1. **Hooks:** react to Claude Code events and write what they see.
2. **Stored state:** holds the data the hooks write.
3. **Panel:** shows the stored state in the UI.

### Layer 1: Hooks
{: .layer-heading}

- **Search hook:** When Claude runs a search, the Claude Code mod records the question and the links in the results. It sends each link to the Google Safe Browsing API to check whether Google has flagged it as malware, phishing, or a deceptive site.
- **`WebFetch` hook:** Before Claude reads a page, the Claude Code mod waits for that page's result from the API. If the API flags the page, the page is blocked. Otherwise, Claude reads the page and can use it in the answer.
- **`turn.complete` hook:** When Claude finishes answering, the Claude Code mod works out which pages the answer came from and marks them as used.

Here's how the `WebFetch` hook decides whether Claude can read a page:

<figure class="diagram">
<svg viewBox="0 0 700 180" role="img" aria-label="The WebFetch hook waits for the Google Safe Browsing API result. Flagged pages are blocked; other pages are read by Claude." class="diagram">
  <defs>
    <marker id="ah-webfetch" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="65" width="150" height="50" rx="3" class="box"></rect>
  <text x="75" y="87" text-anchor="middle" class="label-strong">Claude wants</text>
  <text x="75" y="103" text-anchor="middle" class="label-small">to read a page</text>
  <rect x="200" y="65" width="180" height="50" rx="3" class="box-accent"></rect>
  <text x="290" y="87" text-anchor="middle" class="label-strong">WebFetch hook</text>
  <text x="290" y="103" text-anchor="middle" class="label-small">waits for API result</text>
  <rect x="480" y="10" width="200" height="50" rx="3" class="box"></rect>
  <text x="580" y="32" text-anchor="middle" class="label-strong">Blocked</text>
  <text x="580" y="48" text-anchor="middle" class="label-small">tagged amber</text>
  <rect x="480" y="120" width="200" height="50" rx="3" class="box"></rect>
  <text x="580" y="142" text-anchor="middle" class="label-strong">Claude reads the page</text>
  <text x="580" y="158" text-anchor="middle" class="label-small">tagged green</text>
  <path d="M150,90 L200,90" class="arrow" marker-end="url(#ah-webfetch)"></path>
  <path d="M380,82 L480,35" class="arrow" marker-end="url(#ah-webfetch)"></path>
  <path d="M380,98 L480,145" class="arrow" marker-end="url(#ah-webfetch)"></path>
  <text x="425" y="46" text-anchor="middle" class="label-muted">flagged</text>
  <text x="425" y="146" text-anchor="middle" class="label-muted">not flagged</text>
</svg>
<figcaption>Claude only reads a page after the Google Safe Browsing API has checked it.</figcaption>
</figure>

Here's the check in action. I asked Claude when to hike Mount Fuji. Once the Google Safe Browsing API has checked a search result, the panel adds a green **Safe source** label next to it. Claude then fetches the cleared pages:

<figure class="media-frame">
  <div class="media-frame-window">
    <div class="media-frame-bar"><span class="media-frame-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="media-frame-title">Safety check</span></div>
    <img src="/assets/images/web-activity/30-60.gif" alt="The web activity panel listing nine search results about hiking Mount Fuji, each with a green Safe source label, while Claude fetches pages">
  </div>
  <figcaption>Each search result gets a Safe source label once the Google Safe Browsing API has checked it.</figcaption>
</figure>

### Layer 2: Stored state
{: .layer-heading}

The Claude Code mod keeps two kinds of stored state:

- **Session state** holds the current research and survives a reload of the Claude Code mod. It keeps one entry per search, page, or visit, with its status, results, sections, safety result, and a used flag. A separate `questions` list holds the prompts. When a new question starts its first lookup, the Claude Code mod clears the earlier research.
- **Persistent settings** live and last across sessions. They record whether the panel opens by itself and whether source tracking is on.

### Layer 3: Panel
{: .layer-heading}

The panel listens to the `ui.render` hook. Each time it renders, it reads the stored state; it never keeps its own copy. For each page Claude opened, it shows the Google Safe Browsing API result and whether the answer used the page.

Each page gets a color tag:

- **Green:** safe
- **Blue:** used in the answer
- **Amber:** unsafe

This is how I verify the answer. Once Claude finishes, the pages it drew from get a blue **Used in answer** label. Under each page, the panel lists the sections Claude read, marked with **§**, so I can see exactly where each piece of the answer came from:

<figure class="media-frame">
  <div class="media-frame-window">
    <div class="media-frame-bar"><span class="media-frame-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="media-frame-title">Source verification</span></div>
    <img src="/assets/images/web-activity/120-150.gif" alt="The web activity panel after Claude answers: pages used in the answer carry a blue Used in answer label, and each page lists the sections Claude read from it">
  </div>
  <figcaption>Blue labels mark the pages used in the answer, and the § entries show which section of each page the content came from.</figcaption>
</figure>

### Design considerations

- **Safety checks don't slow down search.** The API check on search results runs in the background, so searches aren't delayed. If the API can't be reached, pages still open.
- **One writer, one reader.** Only the hooks write the stored state, and the panel only reads it. With a single writer, the data can't end up inconsistent.

---

## Setting up the Claude Code mod

### Registering the Claude Code mod

For local development, I register the Claude Code mod in a local marketplace file, `marketplace.json`:

```json
{
  "name": "claude-mods",
  "owner": { "name": "Karthik" },
  "plugins": [
    { "name": "web-activity", "source": "./web-activity", "description": "Shows the searches Claude runs and the web pages it opens, live above the prompt" }
  ]
}
```

The Claude Code mod's own `plugin.json` holds its version. Its `userConfig` block asks for an optional Google Safe Browsing API key:

```json
{
  "name": "web-activity",
  "version": "0.12.0",
  "description": "Shows the searches Claude runs and the web pages it opens, in a panel that opens when Claude searches",
  "types": "./types/index.d.ts",
  "userConfig": {
    "safeBrowsingApiKey": {
      "type": "string",
      "title": "Google Safe Browsing API key",
      "description": "Checks search results and pages against Google Safe Browsing; flagged pages are blocked",
      "sensitive": true,
      "required": false
    }
  }
}
```

### Installing the Claude Code mod

Add the local marketplace to Claude Code:

```
/plugin marketplace add PATH_TO_CLAUDE_CODE_MODS_FOLDER
```

### Updating the Claude Code mod after a change

Claude Code runs an installed copy of the Claude Code mod, not the source folder. So after every change:

<figure class="diagram">
<svg viewBox="0 0 700 100" role="img" aria-label="Updating the Claude Code mod: edit the source folder, bump the version and run update to refresh the installed copy, then restart Claude Code" class="diagram">
  <defs>
    <marker id="ah-update" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="30" width="180" height="50" rx="3" class="box"></rect>
  <text x="90" y="52" text-anchor="middle" class="label-strong">Source folder</text>
  <text x="90" y="68" text-anchor="middle" class="label-small">claude-mods/web-activity</text>
  <rect x="260" y="30" width="180" height="50" rx="3" class="box"></rect>
  <text x="350" y="52" text-anchor="middle" class="label-strong">Installed copy</text>
  <text x="350" y="68" text-anchor="middle" class="label-small">~/.claude/plugins/cache</text>
  <rect x="520" y="30" width="180" height="50" rx="3" class="box-accent"></rect>
  <text x="610" y="52" text-anchor="middle" class="label-strong">Claude Code</text>
  <text x="610" y="68" text-anchor="middle" class="label-small">runs the installed copy</text>
  <path d="M180,55 L260,55" class="arrow" marker-end="url(#ah-update)"></path>
  <path d="M440,55 L520,55" class="arrow" marker-end="url(#ah-update)"></path>
  <text x="220" y="20" text-anchor="middle" class="label-small">bump + update</text>
  <text x="480" y="20" text-anchor="middle" class="label-small">restart</text>
</svg>
<figcaption>Claude Code runs the installed copy, so changes only show up after an update and a restart.</figcaption>
</figure>

Step by step:

1. Edit the source in `claude-mods/web-activity`.
2. Update the version in `plugin.json`.
3. Update the marketplace and the Claude Code mod:
   ```bash
   claude plugin marketplace update claude-mods
   claude plugin update web-activity@claude-mods
   ```
4. Restart Claude Code so it loads the new copy from `~/.claude/plugins/cache/...`.

---

## Conclusion

Every page Claude reads becomes part of its context, so it's worth seeing where those pages come from. This Claude Code mod shows every search and page Claude opens, blocks pages the Google Safe Browsing API has flagged, and marks which sources an answer used.

It won't catch a prompt injection on an ordinary page, but it turns Claude's browsing only to trusted web pages.
