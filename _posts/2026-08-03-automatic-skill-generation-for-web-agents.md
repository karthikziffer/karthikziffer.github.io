---
layout: post
title: "Automatic Skill Generation for Web Agents"
author: "Karthik"
categories: journal
tags: [agentic ai, prompt engineering]
---

<style>
article.article-post h2 {
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin-top: 3.25rem;
  margin-bottom: 1rem;
  color: #1d1d1f;
}
article.article-post h3 {
  font-size: 1.2rem;
  font-weight: 600;
  letter-spacing: -0.005em;
  margin-top: 2.25rem;
  margin-bottom: 0.75rem;
  color: #1d1d1f;
}
article.article-post p {
  color: #333336;
  font-size: 1.05rem;
  line-height: 1.75;
  margin-bottom: 1.35rem;
}
article.article-post > p:first-of-type {
  font-size: 1.22rem;
  line-height: 1.6;
  color: #1d1d1f;
}
article.article-post strong {
  color: #1d1d1f;
}
article.article-post ul {
  margin-bottom: 1.35rem;
  padding-left: 1.3rem;
}
article.article-post ul li {
  margin-bottom: 0.55rem;
  line-height: 1.7;
  color: #333336;
}
article.article-post ul li:last-child {
  margin-bottom: 0;
}
article.article-post hr {
  border: none;
  border-top: 1px solid #e5e5e7;
  margin: 3rem 0;
}
article.article-post pre {
  background: #f6f6f8;
  border: 1px solid #e5e5e7;
  border-radius: 10px;
  padding: 1rem 1.25rem;
  font-size: 0.85rem;
  line-height: 1.65;
}
article.article-post code {
  background: #f2f2f4;
  padding: 0.15em 0.4em;
  border-radius: 5px;
  font-size: 0.85em;
}
article.article-post pre code {
  background: none;
  padding: 0;
}
article.article-post table {
  border: 1px solid #e5e5e7;
  border-radius: 10px;
  overflow: hidden;
  font-size: 0.92rem;
}
article.article-post table th {
  background: #f6f6f8;
  font-weight: 600;
  border-color: #e5e5e7;
}
article.article-post table td {
  border-color: #e5e5e7;
  vertical-align: top;
}
article.article-post img {
  border-radius: 14px;
  border: 1px solid #eee;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
}
article.article-post ol > li {
  margin-bottom: 1.75rem;
  padding-left: 0.25rem;
}
article.article-post ol > li:last-child {
  margin-bottom: 0;
}
.stage-gallery {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin: 1rem 0 0.25rem;
}
.stage-gallery img {
  flex: 1 1 260px;
  max-width: 100%;
  margin: 0;
}
.stage-note {
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #8a8a8f;
  margin: 1.1rem 0 0.6rem;
}
.detour-note {
  display: flex;
  gap: 0.7rem;
  align-items: baseline;
  background: #f9f9fb;
  border-left: 3px solid #03a87c;
  border-radius: 0 10px 10px 0;
  padding: 0.85rem 1.1rem;
  margin: 0.5rem 0 1.75rem;
  font-size: 0.92rem;
  line-height: 1.55;
  color: #4a4a4d;
}
.detour-note .detour-icon {
  flex: none;
  font-size: 1rem;
}
.detour-note a {
  color: #03a87c;
  font-weight: 600;
  text-decoration: none;
}
.detour-note a:hover {
  text-decoration: underline;
}
.kv-calc {
  border: 1px solid #e5e5e7;
  border-radius: 16px;
  background: linear-gradient(180deg, #fcfcfd, #f5f5f7);
  padding: 1.5rem 1.75rem 1.75rem;
  margin: 1.5rem 0 2rem;
}
.kv-diagram-wrap {
  border: 1px solid #e5e5e7;
  border-radius: 16px;
  background: linear-gradient(180deg, #fcfcfd, #f5f5f7);
  padding: 1.25rem 1.5rem 1.5rem;
  margin: 1.5rem 0 2rem;
}
.kv-diagram-wrap svg {
  display: block;
  width: 100%;
  height: auto;
}
.kv-diagram-caption {
  margin: 0.9rem 0 0;
  font-size: 0.85rem;
  line-height: 1.55;
  color: #6b6b70;
  text-align: center;
}
.kv-calc-title {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #8a8a8f;
  margin: 0 0 1rem;
}
.kv-calc-preset {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-bottom: 1.25rem;
}
.kv-calc-preset label {
  font-size: 0.72rem;
  color: #8a8a8f;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.kv-calc-preset select {
  font-size: 0.95rem;
  font-family: inherit;
  font-weight: 600;
  color: #1d1d1f;
  padding: 0.55rem 0.7rem;
  border: 1px solid #d8d8db;
  border-radius: 9px;
  background: #fff;
  width: 100%;
  max-width: 340px;
}
.kv-calc-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 0.9rem;
  padding-top: 1.1rem;
  border-top: 1px dashed #dcdce0;
}
.kv-calc-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.68rem;
  color: #8a8a8f;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.kv-calc-field input,
.kv-calc-field select {
  font-size: 0.92rem;
  font-family: inherit;
  color: #1d1d1f;
  padding: 0.45rem 0.6rem;
  border: 1px solid #d8d8db;
  border-radius: 8px;
  background: #fff;
  width: 100%;
  box-sizing: border-box;
}
.kv-calc-field input:focus,
.kv-calc-field select:focus,
.kv-calc-preset select:focus {
  outline: none;
  border-color: #03a87c;
  box-shadow: 0 0 0 3px rgba(3, 168, 124, 0.14);
}
.kv-calc-formula {
  margin: 1.1rem 0 0;
  font-size: 0.78rem;
  color: #8a8a8f;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.kv-calc-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 0.9rem;
}
.kv-calc-stat {
  flex: 1 1 150px;
  background: #fff;
  border: 1px solid #e5e5e7;
  border-radius: 12px;
  padding: 0.8rem 1rem;
}
.kv-calc-stat .kv-calc-stat-label {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #8a8a8f;
  font-weight: 700;
  margin-bottom: 0.3rem;
}
.kv-calc-stat .kv-calc-stat-value {
  font-size: 1.15rem;
  font-weight: 700;
  color: #1d1d1f;
}
.kv-calc-stat.kv-calc-total {
  background: #03a87c;
  border-color: #03a87c;
}
.kv-calc-stat.kv-calc-total .kv-calc-stat-label {
  color: rgba(255, 255, 255, 0.75);
}
.kv-calc-stat.kv-calc-total .kv-calc-stat-value {
  color: #fff;
  font-size: 1.3rem;
}
</style>

Agents are usually taught with a `skills.md` file that someone writes by hand. I wanted to test a different idea: can that knowledge be generated automatically instead, in a way that's testable and repeatable?

To find out, I built an open source project called [agentic-knowledge-transfer](https://github.com/karthikziffer/agentic-knowledge-transfer). It records a user performing a task on a website, then generates `skills.md` from what it observed, instead of a person writing it by hand. The agent runs on Qwen2.5-VL, a local open-weights model, inferenced through Ollama instead of a hosted foundation model. That also tests something else: whether a local model, given the right harness, can act as an agent on its own.


---

## Overview

Concretely, here's the pipeline:

<div class="kv-diagram-wrap" markdown="0">
  <svg viewBox="0 0 760 240" role="img" aria-labelledby="pipeline-diagram-title">
    <title id="pipeline-diagram-title">Block diagram of the pipeline: Setup creates a project and skill, Record captures a user performing a sample task, Generate summarizes it into skills.md, Verify and Expand tests it live and branches into alternative tasks to broaden skills.md, and Execute lets the agent complete new prompts using only skills.md.</title>
    <defs>
      <marker id="kv3-arrow-muted" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,4 L0,8 z" fill="#b3b3b8"></path>
      </marker>
    </defs>

    <text x="380" y="22" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10.5" letter-spacing="0.06em" fill="#8a8a8f">PIPELINE OVERVIEW</text>

    <!-- connecting arrows -->
    <g stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#kv3-arrow-muted)">
      <path d="M147,125 L166,125"></path>
      <path d="M295,125 L314,125"></path>
      <path d="M443,125 L462,125"></path>
      <path d="M591,125 L610,125"></path>
    </g>

    <!-- Box 1: Setup -->
    <g>
      <rect x="20" y="50" width="127" height="150" rx="12" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <circle cx="38" cy="68" r="11" fill="#03a87c"></circle>
      <text x="38" y="72" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" font-weight="700" fill="#fff">1</text>
      <text x="83.5" y="100" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="700" fill="#1d1d1f">Setup</text>
      <text x="83.5" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">new project</text>
      <text x="83.5" y="138" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">skill + start URL</text>
    </g>

    <!-- Box 2: Record -->
    <g>
      <rect x="168" y="50" width="127" height="150" rx="12" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <circle cx="186" cy="68" r="11" fill="#03a87c"></circle>
      <text x="186" y="72" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" font-weight="700" fill="#fff">2</text>
      <text x="231.5" y="100" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="700" fill="#1d1d1f">Record</text>
      <text x="231.5" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">user performs</text>
      <text x="231.5" y="138" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">the sample task</text>
    </g>

    <!-- Box 3: Generate -->
    <g>
      <rect x="316" y="50" width="127" height="150" rx="12" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <circle cx="334" cy="68" r="11" fill="#03a87c"></circle>
      <text x="334" y="72" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" font-weight="700" fill="#fff">3</text>
      <text x="379.5" y="100" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="700" fill="#1d1d1f">Generate</text>
      <text x="379.5" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">Qwen2.5-VL</text>
      <text x="379.5" y="138" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">summarizes the flow</text>
      <text x="379.5" y="156" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10" font-weight="600" fill="#03a87c">skills.md</text>
    </g>

    <!-- Box 4: Verify & Expand -->
    <g>
      <rect x="464" y="50" width="127" height="150" rx="12" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <circle cx="482" cy="68" r="11" fill="#03a87c"></circle>
      <text x="482" y="72" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" font-weight="700" fill="#fff">4</text>
      <text x="527.5" y="96" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="700" fill="#1d1d1f">Verify &amp;</text>
      <text x="527.5" y="111" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="700" fill="#1d1d1f">Expand</text>
      <text x="527.5" y="130" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">replay live to verify</text>
      <text x="527.5" y="146" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">explore alternatives</text>
      <text x="527.5" y="166" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10" font-weight="600" fill="#03a87c">skills.md</text>
    </g>

    <!-- Box 5: Execute -->
    <g>
      <rect x="612" y="50" width="127" height="150" rx="12" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <circle cx="630" cy="68" r="11" fill="#03a87c"></circle>
      <text x="630" y="72" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" font-weight="700" fill="#fff">5</text>
      <text x="675.5" y="100" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="700" fill="#1d1d1f">Execute</text>
      <text x="675.5" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">new prompt in</text>
      <text x="675.5" y="140" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10" font-weight="600" fill="#03a87c">skills.md only</text>
    </g>

    <text x="380" y="224" text-anchor="middle" font-family="-apple-system, sans-serif" font-style="italic" font-size="10.5" fill="#8a8a8f">Steps 1 to 4 run once per skill; step 5 runs on any new prompt, using only what's already been learned.</text>
  </svg>
</div>

1. Create a project with a webpage URL.

   ![Dashboard with the New Project dialog open](/assets/images/agentic-knowledge-transfer/create-project-dialog.png)

   Then create a task for a skill: name it and give it a start URL, the page the task begins from.

   ![Naming the skill and setting its start URL](/assets/images/agentic-knowledge-transfer/create-skill-dialog.png)

   Both steps of this dialog sit on top of the dashboard, which already tracks aggregate stats per project: total runs, success rate, average run duration, average steps per run. So the effect of every new skill is visible immediately once it's recorded. Clicking "Create & start session" on the second step closes the dialog and drops straight into a live recording session against the start URL.

2. The user performs a sample task on the website. This is recorded along with the accessibility tree, DOM, DOM diffs, and a screenshot of each action.
3. Qwen2.5-VL summarizes the actions performed, and that summary is used to generate a `skills.md` file.

This is the first stage: generating `skills.md` from observed actions instead of a user writing it by hand.

The generated `skills.md` is then tested live against the website to confirm it can reproduce the user's actions exactly.

Once that's verified, the next stage is generating similar tasks on the same website. A depth parameter controls how many alternative tasks get explored. This is essentially telling the agent to learn the website and propose other plausible tasks it could perform. Once these alternatives are generated, a more detailed `skills.md` file can be built that captures the website end to end, giving broader coverage.

Once a website's `skills.md` is available, the final stage is handing the agent a prompt to perform a specific task on that website. The agent relies solely on the website's `skills.md` to complete it.


---

## Context Limitations

<div class="detour-note" markdown="0"><span class="detour-icon">&#9679;</span><span>This section is a technical detour on how an LLM's context and memory work. Skip to <a href="#context-to-knowledge-graph">Context to Knowledge Graph</a> if you just want the project's core idea.</span></div>

Before going further, it's worth explaining why I don't just let the agent rely on the model's context window to hold this knowledge. This project generates a lot of visual and textual artifacts per task, including screenshots, DOM, DOM diffs, and accessibility trees, so I wanted to understand how much of that an LLM's context can actually retain, and for how long.

That raised a few questions: What is context length? When new information comes in, what happens to the older information? I'm using Qwen2.5-VL (`qwen2.5vl:latest`, 8.3B, Q4_K_M, vision) as the completion model, inferenced through Ollama. Here's a brief rundown of how context is managed.

### Context Management

When running Qwen2.5-VL through Ollama, a parameter called `num_ctx` sets the maximum context length allocated for a session. Memory for the KV cache is allocated upfront for the full `num_ctx`, so setting it to 32768, for example, reserves space for 32K tokens' worth of KV vectors even if the conversation is only 500 tokens long.

When context is exceeded in subsequent turns, older tokens get dropped from the front once the limit is hit. Qwen2.5-VL also keeps a prompt cache per session: if the next request shares a prefix with the previous one (say, the same system prompt plus conversation history), it can skip recomputing KV for that prefix, which makes the request faster.

### The KV Cache

Context is the information the model builds up as it receives more input about the data or task. It's computed by the attention mechanism and held in the KV cache. The model's weights themselves are static; when a prompt comes in, the runtime maintains a separate KV cache that grows as tokens are processed.

- Each transformer layer computes a Key and Value vector per token during attention.
- Instead of recomputing these for every prior token at each new generation step, they're cached in GPU/CPU memory.
- Each new token only needs to compute its own KV, then attends over all cached KV from previous tokens.

<div class="kv-diagram-wrap" markdown="0">
  <svg viewBox="0 0 720 400" role="img" aria-labelledby="kv-diagram-title">
    <title id="kv-diagram-title">Diagram of KV cache formation during autoregressive decoding: token 4 computes its own Key/Value, writes them to the cache, and its Query attends over all cached Key/Value pairs including tokens 1-3, which were cached in earlier steps.</title>
    <defs>
      <marker id="kv-arrow-muted" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,4 L0,8 z" fill="#b3b3b8"></path>
      </marker>
      <marker id="kv-arrow-teal" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,4 L0,8 z" fill="#03a87c"></path>
      </marker>
    </defs>

    <text x="360" y="20" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10.5" letter-spacing="0.06em" fill="#8a8a8f">DECODING STEP 4: ONE ATTENTION LAYER</text>

    <!-- Token row -->
    <g font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="13" font-weight="600">
      <rect x="40" y="40" width="64" height="36" rx="9" fill="#fff" stroke="#d8d8db"></rect>
      <text x="72" y="63" text-anchor="middle" fill="#1d1d1f">t1</text>
      <rect x="136" y="40" width="64" height="36" rx="9" fill="#fff" stroke="#d8d8db"></rect>
      <text x="168" y="63" text-anchor="middle" fill="#1d1d1f">t2</text>
      <rect x="232" y="40" width="64" height="36" rx="9" fill="#fff" stroke="#d8d8db"></rect>
      <text x="264" y="63" text-anchor="middle" fill="#1d1d1f">t3</text>
      <rect x="328" y="40" width="64" height="36" rx="9" fill="rgba(3,168,124,0.10)" stroke="#03a87c" stroke-width="1.5"></rect>
      <text x="360" y="63" text-anchor="middle" fill="#03a87c">t4</text>
      <rect x="424" y="40" width="64" height="36" rx="9" fill="none" stroke="#d8d8db" stroke-dasharray="4 3" opacity="0.6"></rect>
      <text x="456" y="63" text-anchor="middle" fill="#b3b3b8" opacity="0.8">t5</text>
    </g>
    <text x="360" y="30" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#03a87c" font-weight="600">new</text>

    <!-- t4 into attention box -->
    <path d="M360,76 L360,98" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#kv-arrow-muted)"></path>
    <rect x="290" y="100" width="140" height="34" rx="8" fill="#f6f6f8" stroke="#d8d8db"></rect>
    <text x="360" y="121" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#4a4a4d">attention (layer i)</text>

    <!-- attends-over fan, drawn behind chips/cells -->
    <g stroke="#03a87c" stroke-width="1.1" fill="none" opacity="0.4" stroke-dasharray="3 3">
      <path d="M315,178 Q 190,225 72,278"></path>
      <path d="M315,178 Q 255,220 168,278"></path>
      <path d="M315,178 Q 300,215 264,278"></path>
      <path d="M315,178 Q 345,220 360,278"></path>
    </g>

    <!-- Q4 K4 V4 chips -->
    <g font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11.5" font-weight="600">
      <path d="M315,134 L315,154" stroke="#8a8a8f" stroke-width="1.2" marker-end="url(#kv-arrow-muted)"></path>
      <rect x="295" y="156" width="40" height="30" rx="7" fill="#f2f2f4" stroke="#c7c7cc" stroke-dasharray="3 2"></rect>
      <text x="315" y="176" text-anchor="middle" fill="#6b6b70">Q4</text>

      <path d="M360,134 L360,154" stroke="#8a8a8f" stroke-width="1.2" marker-end="url(#kv-arrow-muted)"></path>
      <rect x="340" y="156" width="40" height="30" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="360" y="176" text-anchor="middle" fill="#03a87c">K4</text>

      <path d="M405,134 L405,154" stroke="#8a8a8f" stroke-width="1.2" marker-end="url(#kv-arrow-muted)"></path>
      <rect x="385" y="156" width="40" height="30" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="405" y="176" text-anchor="middle" fill="#03a87c">V4</text>
    </g>
    <text x="315" y="206" text-anchor="middle" font-family="-apple-system, sans-serif" font-style="italic" font-size="9.5" fill="#8a8a8f">used once,</text>
    <text x="315" y="217" text-anchor="middle" font-family="-apple-system, sans-serif" font-style="italic" font-size="9.5" fill="#8a8a8f">discarded</text>

    <!-- K4/V4 write into cache -->
    <path d="M360,186 L360,196 Q360,204 366,208 L370,212" stroke="#03a87c" stroke-width="1.3" fill="none" marker-end="url(#kv-arrow-teal)"></path>
    <path d="M405,186 L405,200 Q405,214 380,224 L358,236" stroke="#03a87c" stroke-width="1.3" fill="none" marker-end="url(#kv-arrow-teal)"></path>

    <!-- Cached tokens 1-3 fall straight down (already computed in earlier steps) -->
    <g stroke="#c7c7cc" stroke-width="1.2" stroke-dasharray="4 3">
      <path d="M72,76 L72,278" marker-end="url(#kv-arrow-muted)"></path>
      <path d="M168,76 L168,278" marker-end="url(#kv-arrow-muted)"></path>
      <path d="M264,76 L264,278" marker-end="url(#kv-arrow-muted)"></path>
    </g>
    <text x="130" y="96" text-anchor="middle" font-family="-apple-system, sans-serif" font-style="italic" font-size="10" fill="#b3b3b8">already cached from earlier steps</text>

    <!-- Cache shelf -->
    <rect x="24" y="250" width="464" height="90" rx="14" fill="#fcfcfd" stroke="#e5e5e7"></rect>
    <text x="40" y="268" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="9.5" letter-spacing="0.05em" fill="#8a8a8f">KV CACHE (THIS LAYER)</text>

    <g font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11.5" font-weight="600">
      <rect x="40" y="280" width="64" height="48" rx="8" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <text x="72" y="309" text-anchor="middle" fill="#4a4a4d">K1,V1</text>
      <rect x="136" y="280" width="64" height="48" rx="8" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <text x="168" y="309" text-anchor="middle" fill="#4a4a4d">K2,V2</text>
      <rect x="232" y="280" width="64" height="48" rx="8" fill="#f6f6f8" stroke="#d8d8db"></rect>
      <text x="264" y="309" text-anchor="middle" fill="#4a4a4d">K3,V3</text>
      <rect x="328" y="280" width="64" height="48" rx="8" fill="rgba(3,168,124,0.10)" stroke="#03a87c" stroke-width="1.5"></rect>
      <text x="360" y="309" text-anchor="middle" fill="#03a87c">K4,V4</text>
      <rect x="424" y="280" width="64" height="48" rx="8" fill="none" stroke="#d8d8db" stroke-dasharray="4 3" opacity="0.6"></rect>
      <text x="456" y="309" text-anchor="middle" fill="#c7c7cc">&#8230;</text>
    </g>
    <text x="360" y="270" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#03a87c" font-weight="600">just written</text>

    <text x="360" y="378" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#4a4a4d">Cache grows by one K,V pair per generated token; every later step reuses it, nothing gets recomputed.</text>
  </svg>
</div>

The KV cache scales linearly with context length. A rough formula for the memory cost:

```
2 (K and V) × num_layers × num_heads × head_dim × context_length × bytes_per_value
```

For Qwen2.5-VL's language backbone:

| Parameter | Value |
|---|---|
| num_hidden_layers | 28 |
| num_attention_heads (query heads) | 28 |
| num_key_value_heads (GQA) | 4 |
| head_dim | 128 (3584 hidden / 28 heads) |

**Per-token cache footprint:**

```
elements/token = 2 (K,V) × num_layers × num_kv_heads × head_dim
               = 2 × 28 × 4 × 128
               = 28,672 elements/token
```

At fp16 (2 bytes/value, the default KV cache precision when Qwen2.5-VL is inferenced through Ollama):
`28,672 × 2 bytes = 57,344 bytes/token ≈ 56 KB/token`

At q8_0 quantized KV cache (1 byte/value, optional flag):
`28,672 × 1 byte = 28,672 bytes/token ≈ 28 KB/token`

**Example:** for a 4,096-token session at fp16, that's 28,672 × 4,096 = 117,440,512 elements, or 234,881,024 bytes ≈ 224 MB just for the KV cache.

Try it with other models below, GQA is why Qwen2.5-VL and Llama 3 need a fraction of what Llama 2's full multi-head attention costs.

<div class="kv-calc" markdown="0">
  <p class="kv-calc-title">KV cache size calculator</p>

  <div class="kv-calc-preset">
    <label for="kv-preset">Model preset</label>
    <select id="kv-preset">
      <option value="qwen25vl" selected>Qwen2.5-VL 7B (28L · 4 KV heads · GQA)</option>
      <option value="llama3-8b">Llama 3 8B (32L · 8 KV heads · GQA)</option>
      <option value="mistral-7b">Mistral 7B (32L · 8 KV heads · GQA)</option>
      <option value="llama2-7b">Llama 2 7B (32L · 32 KV heads · MHA, no GQA)</option>
      <option value="custom">Custom</option>
    </select>
  </div>

  <div class="kv-calc-grid">
    <div class="kv-calc-field">
      <label for="kv-layers">Layers</label>
      <input type="number" id="kv-layers" value="28" min="1">
    </div>
    <div class="kv-calc-field">
      <label for="kv-heads">KV heads</label>
      <input type="number" id="kv-heads" value="4" min="1">
    </div>
    <div class="kv-calc-field">
      <label for="kv-headdim">Head dim</label>
      <input type="number" id="kv-headdim" value="128" min="1">
    </div>
    <div class="kv-calc-field">
      <label for="kv-ctx">Context length</label>
      <input type="number" id="kv-ctx" value="4096" min="1">
    </div>
    <div class="kv-calc-field">
      <label for="kv-precision">Precision</label>
      <select id="kv-precision">
        <option value="1">q8_0 (1 byte)</option>
        <option value="2" selected>fp16 (2 bytes)</option>
        <option value="4">fp32 (4 bytes)</option>
      </select>
    </div>
  </div>

  <p class="kv-calc-formula" id="kv-calc-formula"></p>

  <div class="kv-calc-stats">
    <div class="kv-calc-stat">
      <div class="kv-calc-stat-label">Elements / token</div>
      <div class="kv-calc-stat-value" id="kv-stat-elements">—</div>
    </div>
    <div class="kv-calc-stat">
      <div class="kv-calc-stat-label">Bytes / token</div>
      <div class="kv-calc-stat-value" id="kv-stat-perbyte">—</div>
    </div>
    <div class="kv-calc-stat kv-calc-total">
      <div class="kv-calc-stat-label">Total KV cache</div>
      <div class="kv-calc-stat-value" id="kv-stat-total">—</div>
    </div>
  </div>
</div>

<script>
(function () {
  var PRESETS = {
    qwen25vl: { layers: 28, heads: 4, headdim: 128 },
    "llama3-8b": { layers: 32, heads: 8, headdim: 128 },
    "mistral-7b": { layers: 32, heads: 8, headdim: 128 },
    "llama2-7b": { layers: 32, heads: 32, headdim: 128 }
  };

  function fmtBytes(bytes) {
    if (bytes < 1024) return bytes.toFixed(0) + " B";
    var kb = bytes / 1024;
    if (kb < 1024) return kb.toFixed(1) + " KB";
    var mb = kb / 1024;
    if (mb < 1024) return mb.toFixed(2) + " MB";
    return (mb / 1024).toFixed(2) + " GB";
  }
  function fmtNum(n) {
    return Math.round(n).toLocaleString();
  }

  var layersEl = document.getElementById("kv-layers");
  var headsEl = document.getElementById("kv-heads");
  var headdimEl = document.getElementById("kv-headdim");
  var ctxEl = document.getElementById("kv-ctx");
  var precisionEl = document.getElementById("kv-precision");
  var presetEl = document.getElementById("kv-preset");

  function applyPreset() {
    var preset = PRESETS[presetEl.value];
    if (!preset) return;
    layersEl.value = preset.layers;
    headsEl.value = preset.heads;
    headdimEl.value = preset.headdim;
    update();
  }

  function markCustom() {
    presetEl.value = "custom";
  }

  function update() {
    var layers = parseFloat(layersEl.value) || 0;
    var heads = parseFloat(headsEl.value) || 0;
    var headdim = parseFloat(headdimEl.value) || 0;
    var ctx = parseFloat(ctxEl.value) || 0;
    var bpv = parseFloat(precisionEl.value) || 0;

    var perTokenElements = 2 * layers * heads * headdim;
    var perTokenBytes = perTokenElements * bpv;
    var totalBytes = perTokenBytes * ctx;

    document.getElementById("kv-calc-formula").innerHTML =
      "2 &times; " + fmtNum(layers) + " &times; " + fmtNum(heads) + " &times; " + fmtNum(headdim) +
      " &times; " + fmtNum(ctx) + " &times; " + fmtNum(bpv) + " bytes";
    document.getElementById("kv-stat-elements").textContent = fmtNum(perTokenElements);
    document.getElementById("kv-stat-perbyte").textContent = fmtBytes(perTokenBytes);
    document.getElementById("kv-stat-total").textContent = fmtBytes(totalBytes);
  }

  presetEl.addEventListener("change", applyPreset);
  [layersEl, headsEl, headdimEl].forEach(function (el) {
    el.addEventListener("input", function () {
      markCustom();
      update();
    });
  });
  [ctxEl, precisionEl].forEach(function (el) {
    el.addEventListener("input", update);
  });

  update();
})();
</script>

### Vision Encoding

This project deals with screenshots of webpages, so image data has to go through the vision model too. An image is split into pixel patches, producing a grid of visual embeddings. Qwen2.5-VL preserves the image's native aspect ratio and resolution, producing a variable number of visual tokens proportional to image size rather than resizing to a fixed shape. Those patches are converted into embeddings, combined with positional encoding, and inserted directly into the input sequence, which is why higher-resolution images consume more tokens.

This calculation only holds for the autoregressive decoder. When an image comes in, it first goes through the vision encoder, which isn't autoregressive. Unlike the decoder, the encoder doesn't generate patch embeddings one at a time, with each new patch attending back to a cache of earlier ones. Instead, all patches are processed together in a single forward pass, attending to each other directly. The output is the complete set of patch embeddings, computed once, in parallel, so there's no need for caching at the encoder stage.

Once the embeddings are handed to the decoder, they enter the autoregressive stage: the decoder computes KV for those image-embedding tokens once, caches them like any other prompt tokens, and reuses that cache going forward. If the same image is sent again, though, the encoder itself doesn't cache anything; it recomputes from scratch rather than reusing prior work.

<div class="kv-diagram-wrap" markdown="0">
  <svg viewBox="0 0 640 560" role="img" aria-labelledby="vision-diagram-title">
    <title id="vision-diagram-title">Diagram of vision encoding: a screenshot is split into patches and processed once in parallel by the vision encoder with no KV cache, the resulting visual tokens merge with text tokens into one input sequence, and only once that sequence reaches the autoregressive decoder does it get a KV cache, written once and reused thereafter.</title>
    <defs>
      <marker id="kv2-arrow-muted" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,4 L0,8 z" fill="#b3b3b8"></path>
      </marker>
      <marker id="kv2-arrow-teal" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,4 L0,8 z" fill="#03a87c"></path>
      </marker>
    </defs>

    <text x="320" y="20" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10.5" letter-spacing="0.06em" fill="#8a8a8f">SCREENSHOT &#8594; VISION ENCODER &#8594; DECODER</text>

    <!-- Patch grid -->
    <rect x="220" y="40" width="200" height="110" rx="6" fill="#f6f6f8" stroke="#d8d8db"></rect>
    <g stroke="#e5e5e7" stroke-width="1">
      <line x1="253" y1="40" x2="253" y2="150"></line>
      <line x1="287" y1="40" x2="287" y2="150"></line>
      <line x1="320" y1="40" x2="320" y2="150"></line>
      <line x1="353" y1="40" x2="353" y2="150"></line>
      <line x1="387" y1="40" x2="387" y2="150"></line>
      <line x1="220" y1="77" x2="420" y2="77"></line>
      <line x1="220" y1="113" x2="420" y2="113"></line>
    </g>
    <text x="320" y="166" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" fill="#8a8a8f">split into patches, native resolution &amp; aspect ratio preserved</text>

    <path d="M320,178 L320,198" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#kv2-arrow-muted)"></path>

    <!-- Vision encoder -->
    <rect x="190" y="200" width="260" height="64" rx="10" fill="#f2f2f4" stroke="#c7c7cc" stroke-dasharray="5 3"></rect>
    <text x="320" y="224" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="700" fill="#1d1d1f">Vision Encoder</text>
    <text x="320" y="240" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">all patches attend to each other in one parallel pass</text>
    <text x="320" y="253" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">nothing autoregressive here, no KV cache</text>

    <text x="320" y="280" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" fill="#8a8a8f">visual + text tokens merge into one input sequence</text>
    <path d="M320,292 L320,312" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#kv2-arrow-muted)"></path>

    <!-- Merged token row: text tokens + visual tokens -->
    <g font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10.5" font-weight="600">
      <rect x="161" y="314" width="30" height="30" rx="7" fill="#fff" stroke="#d8d8db"></rect>
      <text x="176" y="334" text-anchor="middle" fill="#1d1d1f">t1</text>
      <rect x="197" y="314" width="30" height="30" rx="7" fill="#fff" stroke="#d8d8db"></rect>
      <text x="212" y="334" text-anchor="middle" fill="#1d1d1f">t2</text>
      <rect x="233" y="314" width="30" height="30" rx="7" fill="#f2f2f4" stroke="#c7c7cc"></rect>
      <text x="248" y="334" text-anchor="middle" fill="#6b6b70">v1</text>
      <rect x="269" y="314" width="30" height="30" rx="7" fill="#f2f2f4" stroke="#c7c7cc"></rect>
      <text x="284" y="334" text-anchor="middle" fill="#6b6b70">v2</text>
      <rect x="305" y="314" width="30" height="30" rx="7" fill="#f2f2f4" stroke="#c7c7cc"></rect>
      <text x="320" y="334" text-anchor="middle" fill="#6b6b70">v3</text>
      <rect x="341" y="314" width="30" height="30" rx="7" fill="#f2f2f4" stroke="#c7c7cc"></rect>
      <text x="356" y="334" text-anchor="middle" fill="#6b6b70">v4</text>
      <rect x="377" y="314" width="30" height="30" rx="7" fill="#f2f2f4" stroke="#c7c7cc"></rect>
      <text x="392" y="334" text-anchor="middle" fill="#6b6b70">v5</text>
      <rect x="413" y="314" width="30" height="30" rx="7" fill="none" stroke="#d8d8db" stroke-dasharray="3 2"></rect>
      <text x="428" y="334" text-anchor="middle" fill="#c7c7cc">&#8230;</text>
      <rect x="449" y="314" width="30" height="30" rx="7" fill="#f2f2f4" stroke="#c7c7cc"></rect>
      <text x="464" y="334" text-anchor="middle" fill="#6b6b70">vn</text>
    </g>

    <path d="M320,344 L320,366" stroke="#8a8a8f" stroke-width="1.3" marker-end="url(#kv2-arrow-muted)"></path>

    <!-- Decoder -->
    <rect x="170" y="366" width="300" height="64" rx="10" fill="#f6f6f8" stroke="#d8d8db"></rect>
    <text x="320" y="390" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="700" fill="#1d1d1f">Decoder (autoregressive)</text>
    <text x="320" y="406" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">computes K,V once per token, image tokens included</text>
    <text x="320" y="419" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#6b6b70">then reused every later step, just like text tokens</text>

    <text x="320" y="446" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="#03a87c">written to KV cache, reused every later step</text>
    <path d="M320,458 L320,478" stroke="#03a87c" stroke-width="1.3" marker-end="url(#kv2-arrow-teal)"></path>

    <!-- Cache row -->
    <g font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10.5" font-weight="600">
      <rect x="161" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="176" y="501" text-anchor="middle" fill="#03a87c">t1</text>
      <rect x="197" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="212" y="501" text-anchor="middle" fill="#03a87c">t2</text>
      <rect x="233" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="248" y="501" text-anchor="middle" fill="#03a87c">v1</text>
      <rect x="269" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="284" y="501" text-anchor="middle" fill="#03a87c">v2</text>
      <rect x="305" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="320" y="501" text-anchor="middle" fill="#03a87c">v3</text>
      <rect x="341" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="356" y="501" text-anchor="middle" fill="#03a87c">v4</text>
      <rect x="377" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="392" y="501" text-anchor="middle" fill="#03a87c">v5</text>
      <rect x="413" y="480" width="30" height="34" rx="7" fill="none" stroke="#03a87c" stroke-dasharray="3 2" opacity="0.6"></rect>
      <text x="428" y="501" text-anchor="middle" fill="#03a87c" opacity="0.6">&#8230;</text>
      <rect x="449" y="480" width="30" height="34" rx="7" fill="rgba(3,168,124,0.10)" stroke="#03a87c"></rect>
      <text x="464" y="501" text-anchor="middle" fill="#03a87c">vn</text>
    </g>

    <text x="320" y="530" text-anchor="middle" font-family="-apple-system, sans-serif" font-style="italic" font-size="10.5" fill="#8a8a8f">If the same image is sent again, the encoder still recomputes from scratch.</text>
    <text x="320" y="545" text-anchor="middle" font-family="-apple-system, sans-serif" font-style="italic" font-size="10.5" fill="#8a8a8f">Only the decoder's KV cache persists.</text>
  </svg>
</div>


---

## Context to Knowledge Graph

All of this points to the same conclusion: the KV cache is a good mechanism for a single session, but it's ephemeral, bound to one model, and expensive to grow. It's the wrong place to store knowledge that should persist and transfer across tasks, sessions, or even models.

That's the core idea behind this project: store the context knowledge as a knowledge graph in Memgraph, built from tasks and their generated alternatives, and reuse it on new, unseen tasks. The hypothesis is that once an agent learns the foundational structure of a website, other tasks on that same site become easier to perform autonomously.

This reframes the problem: instead of relying on the model's in-memory context, tasks become entries in a knowledge graph. That knowledge becomes transferable across models and no longer depends on any single model's context window.


---

## Stages

The workflow is broken into stages. First, a project is created for a website. Then a skill is built for it: a skill is just the set of actions needed to perform one task on that website.

1. **Teaching**: the user performs a sample task on the website. In the recording above, that's just clicking through to an article on Hacker News: the cursor lands on "Don't stop early: Case-folding source code at memory speed" and the page navigates to the GitHub blog post behind it. Every action like this is logged as its own step in a run-flow timeline (session started, clicked link, session finished), each tagged `DONE`, timestamped against the previous step (`+6.8s`, `+4.0s`), and paired with a screenshot when one exists. Steps with no visual change, like the click itself, fall back to a `DOM diff` instead of a placeholder screenshot.

   <div class="stage-gallery" markdown="0">
     <img src="/assets/images/agentic-knowledge-transfer/stage-1-teaching.gif" alt="Recording a sample task on the website">
     <img src="/assets/images/agentic-knowledge-transfer/stage-1-run-flow.png" alt="Recorded run flow timeline">
   </div>

2. **Summarization**: the agent takes the screenshots and actions from that task and summarizes the complete flow. This happens in a dedicated tab of the same workspace (alongside `skills.md`, testing, and alternative planning), and the output is attributed to the model and timestamped, e.g. `qwen2.5vl:latest · 8/4/2026, 11:16:40 PM`, with a `Regenerate` button next to it. The step-by-step view underneath shows the model's reasoning per action: it explains when it skipped vision analysis outright, for instance the "session started" and "session finished" markers aren't user actions, so the model reuses the recorded description rather than inventing one, and for the click step it has no screenshot to work from, so it falls back to the recorded step type and the Playwright locator (`page.getByRole('link', { name: ..., exact: true })`) instead of guessing.

   <div class="stage-gallery" markdown="0">
     <img src="/assets/images/agentic-knowledge-transfer/stage-2-flow-summary.png" alt="AI-generated flow summary">
     <img src="/assets/images/agentic-knowledge-transfer/stage-2-step-by-step.png" alt="Step-by-step model reasoning for each action">
   </div>

3. **Skill generation**: from that summary, a `skills.md` file is created for the task.

   <p class="stage-note" markdown="0">Generated <code>skills.md</code> for the Hacker News task</p>

   <div markdown="0"><pre><code>---
   name: hacker-news
   description: The session began with navigating to Hacker News.
   ---

   # hacker news

   The session began with navigating to Hacker News. The user then clicked on
   an article titled "Don't stop early: Case-folding source code at memory
   speed," which seemed intriguing.

   ## Start

   Navigate to: https://news.ycombinator.com/

   ## Steps

   1. Session started
   2. Clicked link "Don't stop early: Case-folding source code at memory speed"
      Target: page.getByRole('link', { name: 'Don\'t stop early: Case-folding source code at memory speed', exact: true })
   3. Session finished

   ## Expected outcome

   The session succeeded as it allowed the user to explore a specific topic on
   Hacker News.

   ---</code></pre></div>

4. **Testing the skill**: the generated `skills.md` is tested live against the website to confirm it can reproduce the user's task exactly. A `LIVE VIEW` panel replays the browser in real time while a checklist on the side ticks off each step as it happens: `Replay started`, then the click itself tagged `AUTO` to mark it as agent-driven rather than user-recorded, then `Replay finished`. The panel flips from `RUNNING` to `COMPLETED` once the agent lands back on the same GitHub blog post the human did.

   ![Replaying the generated skills.md live against the website](/assets/images/agentic-knowledge-transfer/stage-4-test-skill.gif)

5. **Alternative generation**: `skills.md`, along with the task artifacts (screenshots, DOM, DOM diffs, and accessibility tree for each action), is used to generate alternative tasks that are possible on the same website. A depth parameter controls how many steps are explored per alternative. The planner branches out from the recorded step as a tree: in this run it found two candidate plans, each going two links deep, with a "Run this plan" action on every leaf. A `GENERATED SO FAR` log below the tree tracks completed runs with their full path, for example `link "new" → link "We finally learned to center a div, then browsers added sidebars" → link "writing"`, each marked `COMPLETED` with a link back to the recording. The two runs shown here both start from Hacker News' `/new` page: one ends on a blog post about CSS centering, the other on a Substack video post by Ethan Smith.

   ![Planning alternative tasks branching from a recorded step](/assets/images/agentic-knowledge-transfer/stage-5-alternative-generation.png)

   <div class="stage-gallery" markdown="0">
     <img src="/assets/images/agentic-knowledge-transfer/stage-5-alternative-plan-1.gif" alt="Alternative plan 1 running">
     <img src="/assets/images/agentic-knowledge-transfer/stage-5-alternative-plan-2.gif" alt="Alternative plan 2 running">
   </div>

6. **Knowledge graph construction**: an automated pipeline generates these alternatives and writes the resulting task knowledge into Memgraph, forming a reference for future tasks. While it runs, one panel streams a log of every link it tries, marked `DONE` or `ERROR` (`Element not found` shows up for a couple of stale links), while another panel grows the graph live underneath: page screenshots become nodes, and each click that connects two pages becomes an edge between them. Nodes are inspectable: clicking one surfaces an aggregate like `11 actions here, 9 ok, 2 failed` for the Hacker News guidelines page, tallying every attempt across all the alternative runs that passed through it.

   ![Task knowledge being written into the Memgraph knowledge graph](/assets/images/agentic-knowledge-transfer/stage-6-knowledge-graph.gif)

7. **Autonomous execution**: given a prompt, the agent uses the knowledge graph to perform the requested task.

That's the complete goal of the project.


---

## Applications

1. **Automated testing**: within an organization's website, checking all possible options and letting agents run automated tests from just a prompt.
2. **Private, local skill development**: building custom skills for websites that run locally, where data needs to stay private.
3. **A playground for skill optimization**: using the platform to iterate on and refine skills.

Further, this platform could expand to testing agents on multistep tasks.


---

## Conclusion

This is still an early prototype, tested on one website with one simple task. Making it robust across websites needs a lot more work: different DOM structures, dynamic content, logins, anti-bot checks, and layout changes all need handling. The alternative-generation and knowledge graph parts are also still untested at scale. It's far from finished, but it's a good base to keep building on.
