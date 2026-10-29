---
layout: post
title: "LLM as a verifier instead of a judge"
author: "Karthik"
categories: journal
tags: [llm, agentic-ai]
---

Most agentic workflows use an LLM as a judge to decide whether a candidate answer (the answer being evaluated) is correct. The judge rates each candidate answer with a judge score, for example from 0 to 5, but it often gives most answers the same score. When two models both earn a 4 on the same question, the judge score cannot show which one is better:

<figure class="diagram">
<svg viewBox="0 0 700 150" role="img" aria-label="Two models answer the same question. The LLM as a judge gives both candidate answers a judge score of 4, so it cannot show which model is better." class="diagram">
  <defs>
    <marker id="ah-problem" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="10" width="145" height="44" rx="3" class="box"></rect>
  <text x="72.5" y="37" text-anchor="middle" class="label-strong">Model A</text>
  <rect x="185" y="10" width="145" height="44" rx="3" class="box"></rect>
  <text x="257.5" y="37" text-anchor="middle" class="label">Candidate answer</text>
  <rect x="370" y="10" width="145" height="44" rx="3" class="box"></rect>
  <text x="442.5" y="37" text-anchor="middle" class="label">LLM as a judge</text>
  <rect x="555" y="10" width="145" height="44" rx="3" class="box-dashed"></rect>
  <text x="627.5" y="37" text-anchor="middle" class="label-accent">Judge score: 4</text>
  <path d="M145,32 L185,32" class="arrow" marker-end="url(#ah-problem)"></path>
  <path d="M330,32 L370,32" class="arrow" marker-end="url(#ah-problem)"></path>
  <path d="M515,32 L555,32" class="arrow" marker-end="url(#ah-problem)"></path>

  <rect x="0" y="74" width="145" height="44" rx="3" class="box"></rect>
  <text x="72.5" y="101" text-anchor="middle" class="label-strong">Model B</text>
  <rect x="185" y="74" width="145" height="44" rx="3" class="box"></rect>
  <text x="257.5" y="101" text-anchor="middle" class="label">Candidate answer</text>
  <rect x="370" y="74" width="145" height="44" rx="3" class="box"></rect>
  <text x="442.5" y="101" text-anchor="middle" class="label">LLM as a judge</text>
  <rect x="555" y="74" width="145" height="44" rx="3" class="box-dashed"></rect>
  <text x="627.5" y="101" text-anchor="middle" class="label-accent">Judge score: 4</text>
  <path d="M145,96 L185,96" class="arrow" marker-end="url(#ah-problem)"></path>
  <path d="M330,96 L370,96" class="arrow" marker-end="url(#ah-problem)"></path>
  <path d="M515,96 L555,96" class="arrow" marker-end="url(#ah-problem)"></path>

  <text x="700" y="140" text-anchor="end" class="label-muted">same score: which model is better?</text>
</svg>
<figcaption>Both models earn the same judge score, so the judge cannot show which one is better.</figcaption>
</figure>

The [LLM as a verifier](https://llm-as-a-verifier.notion.site/) research, from the work of [Azalia Mirhoseini](https://azaliamirhoseini.com/), addresses this. The evaluating LLM answers with a label, such as A to E, and the verifier reads the probability of every label instead of only the one the LLM picked.

The post makes two contributions:

- **A comparison of the two techniques.** Two small local answer models answer the first 100 GSM8K math questions (grade-school word problems from OpenAI), and Llama 3.1 scores every answer, once as a judge and once as a verifier.
- **A fix for the practical problems.** Reading label probabilities from a generative LLM is error-prone, so I also use a decision model, Nimble, as the verifier.

Each candidate answer is scored only once, so the results do not correct for single-run bias.

In this post:

1. [Why LLM as a judge falls short](#why-llm-as-a-judge-falls-short)
2. [How LLM as a verifier works](#how-llm-as-a-verifier-works)
3. [Problems with reading label probabilities](#problems-with-reading-label-probabilities)
4. [A decision model avoids these problems](#a-decision-model-avoids-these-problems)
5. [The experiment](#the-experiment)
6. [Results](#results)
7. [Conclusion](#conclusion)

---

## Why LLM as a judge falls short

An LLM as a judge reads a candidate answer and returns one judge score, for example from 0 (worst) to 5 (best). Each score is a category. The main challenge is that many answers receive the same judge score, which leaves no signal to tell them apart.

Here are two answers that receive the same judge score for very different reasons:

<figure class="diagram">
<svg viewBox="0 0 700 225" role="img" aria-label="Two answers with probability bars for judge scores 0 to 5. The judge is confident about the first answer and unsure about the second, but both receive a judge score of 4." class="diagram">
  <text x="130" y="20" text-anchor="middle" class="lane-title">Answer 1: confident</text>
  <text x="570" y="20" text-anchor="middle" class="lane-title">Answer 2: unsure</text>

  <rect x="0" y="168.8" width="30" height="1.2" class="box"></rect>
  <rect x="46" y="168.8" width="30" height="1.2" class="box"></rect>
  <rect x="92" y="166.4" width="30" height="3.6" class="box"></rect>
  <rect x="138" y="164.0" width="30" height="6.0" class="box"></rect>
  <rect x="184" y="68.0" width="30" height="102.0" class="box-accent"></rect>
  <rect x="230" y="164.0" width="30" height="6.0" class="box"></rect>
  <text x="15" y="163" text-anchor="middle" class="label-small">0.01</text>
  <text x="61" y="163" text-anchor="middle" class="label-small">0.01</text>
  <text x="107" y="160" text-anchor="middle" class="label-small">0.03</text>
  <text x="153" y="158" text-anchor="middle" class="label-small">0.05</text>
  <text x="199" y="62" text-anchor="middle" class="label-accent">0.85</text>
  <text x="245" y="158" text-anchor="middle" class="label-small">0.05</text>
  <line x1="0" y1="170" x2="260" y2="170" class="lane-rule"></line>
  <text x="15" y="188" text-anchor="middle" class="label-small">0</text>
  <text x="61" y="188" text-anchor="middle" class="label-small">1</text>
  <text x="107" y="188" text-anchor="middle" class="label-small">2</text>
  <text x="153" y="188" text-anchor="middle" class="label-small">3</text>
  <text x="199" y="188" text-anchor="middle" class="label-small">4</text>
  <text x="245" y="188" text-anchor="middle" class="label-small">5</text>
  <text x="130" y="214" text-anchor="middle" class="label-strong">Judge score: 4</text>

  <rect x="440" y="167.6" width="30" height="2.4" class="box"></rect>
  <rect x="486" y="166.4" width="30" height="3.6" class="box"></rect>
  <rect x="532" y="158.0" width="30" height="12.0" class="box"></rect>
  <rect x="578" y="124.4" width="30" height="45.6" class="box"></rect>
  <rect x="624" y="119.6" width="30" height="50.4" class="box-accent"></rect>
  <rect x="670" y="164.0" width="30" height="6.0" class="box"></rect>
  <text x="455" y="162" text-anchor="middle" class="label-small">0.02</text>
  <text x="501" y="160" text-anchor="middle" class="label-small">0.03</text>
  <text x="547" y="152" text-anchor="middle" class="label-small">0.10</text>
  <text x="593" y="118" text-anchor="middle" class="label-small">0.38</text>
  <text x="639" y="114" text-anchor="middle" class="label-accent">0.42</text>
  <text x="685" y="158" text-anchor="middle" class="label-small">0.05</text>
  <line x1="440" y1="170" x2="700" y2="170" class="lane-rule"></line>
  <text x="455" y="188" text-anchor="middle" class="label-small">0</text>
  <text x="501" y="188" text-anchor="middle" class="label-small">1</text>
  <text x="547" y="188" text-anchor="middle" class="label-small">2</text>
  <text x="593" y="188" text-anchor="middle" class="label-small">3</text>
  <text x="639" y="188" text-anchor="middle" class="label-small">4</text>
  <text x="685" y="188" text-anchor="middle" class="label-small">5</text>
  <text x="570" y="214" text-anchor="middle" class="label-strong">Judge score: 4</text>

  <text x="350" y="214" text-anchor="middle" class="label-muted">same judge score</text>
</svg>
<figcaption>The judge is confident about the first answer and torn between 3 and 4 on the second. Both still receive a 4.</figcaption>
</figure>

This creates two problems:

- **Scores cluster together.** When most answers receive a 4, the scores can no longer separate a strong model from a weaker one.
- **A single score hides the details.** One number does not reveal which part of an answer was strong and which part was weak.

Rubrics, used in more advanced versions of the LLM as a judge, solve the second problem. Instead of asking for one overall score, you split the evaluation into smaller sub-questions or gate checks (a rubric) and ask the judge to score each one separately. The result is far more detailed, but every score is still a single category, so the first problem remains: the scores still cluster together.

---

## How LLM as a verifier works

With an LLM as a verifier, the evaluating LLM answers with a label, such as A, B, C, D, or E. In this post, A is the best label and E is the worst.

The key idea is to look past the label the LLM chooses. Every token an LLM generates comes with a log probability, which measures how likely the LLM thought that token was. The verifier reads the log probabilities of all five labels, so it captures not just the LLM's choice but how confident it was.

Each label is also assigned a fixed reward value, spaced evenly from 1 for the best label down to 0 for the worst. These are the values I use in the experiment:

| Label | Reward | Meaning              |
|-------|--------|----------------------|
| A     | 1.00   | Fully correct        |
| B     | 0.75   | Mostly correct       |
| C     | 0.50   | Partially correct    |
| D     | 0.25   | Mostly incorrect     |
| E     | 0.00   | Completely incorrect |

The verifier combines the log probabilities and rewards into the verifier score in four steps:

<figure class="diagram">
<svg viewBox="0 0 700 100" role="img" aria-label="The verifier score: label log probabilities become probabilities, are normalized to sum to 1, and are multiplied by each label's reward and added up" class="diagram">
  <defs>
    <marker id="ah-verifier" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="72.5" y="52" text-anchor="middle" class="label-strong">Log probabilities</text>
  <text x="72.5" y="68" text-anchor="middle" class="label-small">one per label</text>
  <rect x="185" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="257.5" y="52" text-anchor="middle" class="label-strong">Probabilities</text>
  <text x="257.5" y="68" text-anchor="middle" class="label-small">p = e^(log probability)</text>
  <rect x="370" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="442.5" y="52" text-anchor="middle" class="label-strong">Normalized</text>
  <text x="442.5" y="68" text-anchor="middle" class="label-small">add up to 1</text>
  <rect x="555" y="30" width="145" height="50" rx="3" class="box-accent"></rect>
  <text x="627.5" y="52" text-anchor="middle" class="label-strong">Verifier score</text>
  <text x="627.5" y="68" text-anchor="middle" class="label-small">sum of p × reward</text>
  <path d="M145,55 L185,55" class="arrow" marker-end="url(#ah-verifier)"></path>
  <path d="M330,55 L370,55" class="arrow" marker-end="url(#ah-verifier)"></path>
  <path d="M515,55 L555,55" class="arrow" marker-end="url(#ah-verifier)"></path>
  <text x="165" y="20" text-anchor="middle" class="label-small">convert</text>
  <text x="350" y="20" text-anchor="middle" class="label-small">÷ sum</text>
  <text x="535" y="20" text-anchor="middle" class="label-small">× reward</text>
</svg>
<figcaption>The verifier score draws on every label's probability, not just the label the LLM picked.</figcaption>
</figure>

1. Take the log probability of each label.
2. Convert each log probability to a probability.
3. Divide each probability by their sum, so they add up to 1.
4. Multiply each probability by its label's reward, and add the results.

To make this concrete, here is the calculation for an answer the verifier considers likely to be good:

| Label | Log probability | Probability | Normalized | × Reward | Result |
|-------|-----------------|-------------|------------|----------|--------|
| A     | −0.223          | 0.80        | 0.808      | 1.00     | 0.808  |
| B     | −2.303          | 0.10        | 0.101      | 0.75     | 0.076  |
| C     | −3.507          | 0.03        | 0.030      | 0.50     | 0.015  |
| D     | −3.507          | 0.03        | 0.030      | 0.25     | 0.008  |
| E     | −3.507          | 0.03        | 0.030      | 0.00     | 0.000  |
| Sum   |                 | 0.99        | 1.000      |          | 0.907  |

The verifier score is 0.907. A verifier score near 1 means the verifier is confident the answer is good, a score near 0 means it is confident the answer is bad, and a score in the middle means it is undecided.

This is what sets the verifier apart from the judge. A judge score is one category. A verifier score is a continuous number shaped by the probability of every label. Two answers that would both receive a 4 from a judge can still receive different verifier scores, and that difference is exactly the signal the judge throws away.

---

## Problems with reading label probabilities

The method assumes the LLM returns a clean probability for each label. In practice, this rarely holds, and four problems come up.

### Labels are mixed with other tokens

An LLM generates text one token at a time, and each token carries its own probability. Ideally, the output would contain only the label options and their probabilities. In reality, the label is one token among many:

<figure class="diagram">
<svg viewBox="0 0 700 205" role="img" aria-label="Ideal output: only the label options A, B, and C at one position, with probabilities 0.80, 0.10, and 0.03. Actual output: the LLM predicts the tokens A, is, the, best, label, each with its own probability, and the label is only one of them." class="diagram">
  <text x="0" y="18" class="lane-title">Ideal output: label options only</text>
  <rect x="0" y="30" width="100" height="36" rx="3" class="box-accent"></rect>
  <text x="50" y="53" text-anchor="middle" class="label-strong">A</text>
  <text x="50" y="84" text-anchor="middle" class="label-accent">p = 0.80</text>
  <rect x="120" y="30" width="100" height="36" rx="3" class="box-accent"></rect>
  <text x="170" y="53" text-anchor="middle" class="label-strong">B</text>
  <text x="170" y="84" text-anchor="middle" class="label-accent">p = 0.10</text>
  <rect x="240" y="30" width="100" height="36" rx="3" class="box-accent"></rect>
  <text x="290" y="53" text-anchor="middle" class="label-strong">C</text>
  <text x="290" y="84" text-anchor="middle" class="label-accent">p = 0.03</text>
  <text x="360" y="53" class="label-muted">alternatives at one position</text>

  <line x1="0" y1="104" x2="700" y2="104" class="lane-rule"></line>

  <text x="0" y="128" class="lane-title">Actual output</text>
  <rect x="0" y="140" width="100" height="36" rx="3" class="box-accent"></rect>
  <text x="50" y="163" text-anchor="middle" class="label-strong">A</text>
  <rect x="120" y="140" width="100" height="36" rx="3" class="box"></rect>
  <text x="170" y="163" text-anchor="middle" class="label">is</text>
  <rect x="240" y="140" width="100" height="36" rx="3" class="box"></rect>
  <text x="290" y="163" text-anchor="middle" class="label">the</text>
  <rect x="360" y="140" width="100" height="36" rx="3" class="box"></rect>
  <text x="410" y="163" text-anchor="middle" class="label">best</text>
  <rect x="480" y="140" width="100" height="36" rx="3" class="box"></rect>
  <text x="530" y="163" text-anchor="middle" class="label">label</text>
  <rect x="600" y="140" width="100" height="36" rx="3" class="box"></rect>
  <text x="650" y="163" text-anchor="middle" class="label-muted">…</text>
  <text x="50" y="194" text-anchor="middle" class="label-accent">p = 0.80</text>
  <text x="170" y="194" text-anchor="middle" class="label-small">p = 0.91</text>
  <text x="290" y="194" text-anchor="middle" class="label-small">p = 0.88</text>
  <text x="410" y="194" text-anchor="middle" class="label-small">p = 0.62</text>
  <text x="530" y="194" text-anchor="middle" class="label-small">p = 0.95</text>
  <text x="650" y="194" text-anchor="middle" class="label-small">…</text>
</svg>
<figcaption>Every predicted token has its own probability. The verifier needs only the probabilities at the label token's position. The values shown are illustrative.</figcaption>
</figure>

The label probabilities come from the position where the label token appears. At that position, the API (Ollama, in my experiment) also returns the most likely alternative tokens, including the other labels.

The code therefore has to search the generated tokens, find the label token, and read the probabilities at that position. This is also a strong reason to use fixed letter labels instead of numbers. A letter like `A` stands out among the other tokens, while a number like `4` could just as easily come from the math itself.

### Labels can be missing from the returned tokens

When you request log probabilities, the API returns only the most likely tokens at each position, for example the top 20. Any label less likely than those is not returned at all.

A missing label cannot safely be treated as probability 0. Without it, the verifier score for that candidate answer cannot be calculated correctly.

### Probabilities do not add up to 1

The five label probabilities rarely add up to 1, because some of the probability goes to other tokens. In the example above, they add up to 0.99.

Normalization solves this: divide each probability by their sum. The individual values change, but the ratios between labels stay the same, and those ratios are what the verifier score depends on.

### The same label can appear in different forms

An LLM can produce the same label as different tokens, such as `C` and `c`. Each form receives a share of the probability, so counting only one of them underestimates that label. Add the forms together before normalizing.

---

## A decision model avoids these problems

All four problems have the same source: a generative LLM writes free text, and the label is only one token in it. As an alternative to the Llama 3.1 verifier, I extended the LLM as a verifier with a decision model called Nimble.

A decision model does not generate free text. It returns only one of a fixed set of labels, together with the probability of each label. Because no other tokens are generated, the labels cannot be mixed with other text or go missing, and the probabilities always add up to 1. The verifier score is then calculated from these probabilities in the same way.

---

## The experiment

### Setup

I used four models, all running locally with Ollama:

| Role                      | Model                        | Size   |
|---------------------------|------------------------------|--------|
| Answer model              | `openbmb/minicpm5-2b:latest` | 1.6 GB |
| Answer model              | `deepseek-r1:1.5b`           | 1.1 GB |
| Judge and verifier        | `llama3.1:latest`            | 4.9 GB |
| Verifier (decision model) | `nimble:latest`              | 9.3 GB |

Each GSM8K item has a question and a reference answer. The reference answer works through the problem step by step, marks each calculation with `<<…>>`, and ends with the final answer after `####`. Here is a sample item:

```json
{
  "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
  "answer": "Maila read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.\n#### 42"
}
```

The answer models see only the question. The evaluators (the Llama 3.1 judge, the Llama 3.1 verifier, and the Nimble verifier) see only the question and the candidate answer. The reference answer is used only afterwards, to check which candidate answers are correct. For this item, the final answer is 42. (The switch from Julie to Maila is in the original dataset.)

Each answer model answers the same 100 GSM8K questions: the first 100 in the dataset, chosen to test the LLM as a verifier technique. Llama 3.1 then evaluates every candidate answer on three criteria: correctness, reasoning, and consistency. As a judge, it returns a judge score from 0 to 5 for each criterion. As a verifier, it returns a label from A to E. The Nimble verifier then scores the same candidate answers. The full pipeline looks like this:

<figure class="diagram">
<svg viewBox="0 0 700 120" role="img" aria-label="Experiment pipeline: 100 GSM8K questions go to two answer models. Three evaluators score each candidate answer: the Llama 3.1 judge, the Llama 3.1 verifier, and the Nimble verifier." class="diagram">
  <defs>
    <marker id="ah-pipeline" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="30" width="180" height="60" rx="3" class="box"></rect>
  <text x="90" y="56" text-anchor="middle" class="label-strong">GSM8K</text>
  <text x="90" y="72" text-anchor="middle" class="label-small">100 math questions</text>
  <rect x="260" y="30" width="180" height="60" rx="3" class="box"></rect>
  <text x="350" y="52" text-anchor="middle" class="label-strong">Answer models</text>
  <text x="350" y="68" text-anchor="middle" class="label-small">minicpm5-2b</text>
  <text x="350" y="82" text-anchor="middle" class="label-small">deepseek-r1</text>
  <rect x="520" y="30" width="180" height="60" rx="3" class="box-accent"></rect>
  <text x="610" y="52" text-anchor="middle" class="label-strong">Evaluators</text>
  <text x="610" y="68" text-anchor="middle" class="label-small">Llama 3.1: judge, verifier</text>
  <text x="610" y="82" text-anchor="middle" class="label-small">Nimble: verifier</text>
  <path d="M180,60 L260,60" class="arrow" marker-end="url(#ah-pipeline)"></path>
  <path d="M440,60 L520,60" class="arrow" marker-end="url(#ah-pipeline)"></path>
  <text x="220" y="20" text-anchor="middle" class="label-small">question</text>
  <text x="480" y="20" text-anchor="middle" class="label-small">candidate answer</text>
  <text x="610" y="110" text-anchor="middle" class="label-muted">three criteria per answer</text>
</svg>
<figcaption>Three evaluators score every candidate answer: the Llama 3.1 judge, the Llama 3.1 verifier, and the Nimble verifier.</figcaption>
</figure>

### Prompts

The Llama 3.1 judge scores all three criteria in one request and returns a JSON object:

```text
No reference solution is available: check the answer by solving the problem yourself.
First, work through the problem independently and recompute the key numbers.
Then score the generated answer on correctness, reasoning and consistency, each an integer from 0 to 5
(5 = exactly correct / sound / self-consistent, 3 = minor errors, 0 = wrong).
Return ONLY a JSON object: working, correctness, reasoning, consistency, explanation.
```

The Llama 3.1 verifier is asked once per criterion and must answer with a single label, A to E. Its verifier score is calculated from the log probabilities of the five labels, as described above:

```text
No reference solution is available: check the candidate answer by verifying the mathematics yourself.
Evaluate only this criterion: {criterion}: {description}
Assign exactly one score token:
A = fully correct, B = mostly correct, C = partially correct, D = mostly incorrect, E = completely incorrect
Output exactly one uppercase letter, A through E.
```

The Nimble verifier receives the prompt below for each criterion. Its labels are the same five, written as words instead of letters, and its verifier score is calculated from Nimble's probability for each label:

```text
Grade the candidate answer on {criterion}. {description}
No reference solution is available: check the candidate answer by verifying the mathematics yourself.
Levels: completely incorrect < mostly incorrect < partially correct < mostly correct < fully correct
```

---

## Results

I compare the scores from the three evaluators. The charts show how each evaluator's scores are distributed for the two answer models, and the last test checks which evaluator best picks out the correct candidate answer.

### Judge scores cluster at the top

Without the reference answer, the judge gives almost every candidate answer a 5 on all three criteria:

<figure class="media-frame">
  <div class="media-frame-window">
    <div class="media-frame-bar"><span class="media-frame-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="media-frame-title">Judge scores</span></div>
    <img src="/assets/images/llm-as-a-verifier/score_distribution.png" alt="Bar charts of LLM as a judge scores from 0 to 5 for correctness, reasoning, and consistency. For both answer models, about 90 of 100 answers receive a 5 on every criterion.">
  </div>
  <figcaption>Around 90 of 100 answers receive a 5 on every criterion, so the two answer models are almost indistinguishable.</figcaption>
</figure>

The mean judge scores of the two answer models differ by at most 0.10 on any criterion, so the judge cannot tell them apart.

### One judge score becomes a distribution

To see what the verifiers add, I took only the answers the judge scored 5 out of 5 for correctness, and plotted the verifier score each one received:

<figure class="media-frame">
  <div class="media-frame-window">
    <div class="media-frame-bar"><span class="media-frame-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="media-frame-title">Verifier scores on the judge's 5/5 answers</span></div>
    <img src="/assets/images/llm-as-a-verifier/judge5_noref_verifiers_by_model.png" alt="Distributions of verifier correctness scores from the Llama 3.1 verifier and the Nimble verifier, for only the answers the judge scored 5 out of 5. For deepseek-r1, 92 answers with 24 wrong; for minicpm5-2b, 90 answers with 9 wrong. The scores spread from below 0.5 to above 0.999.">
  </div>
  <figcaption>Every answer here received the same judge score of 5. Both verifiers spread them across a wide range of verifier scores.</figcaption>
</figure>

To the judge, all of these answers are identical. The verifiers spread the same answers across a wide range: the Llama 3.1 verifier from about 0.5 to above 0.999, and the Nimble verifier from about 0.25. Each tick under a curve is one answer, and the horizontal axis is stretched near 1 so that small differences between high scores stay visible.

This matters because a perfect judge score does not guarantee a correct candidate answer. Of the 92 deepseek-r1 answers the judge scored 5, 24 are wrong. For minicpm5-2b, 9 of 90 are wrong. A judge score of 5 cannot separate these wrong answers from the correct ones, while the verifiers at least spread them out.

### The verifier separates the answer models

The Llama 3.1 verifier uses the same model as the judge, but spreads the scores out. In these charts, each answer's verifier score is rounded to the nearest label:

<figure class="media-frame">
  <div class="media-frame-window">
    <div class="media-frame-bar"><span class="media-frame-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="media-frame-title">Verifier scores: Llama 3.1</span></div>
    <img src="/assets/images/llm-as-a-verifier/reward_distribution.png" alt="Bar charts of LLM as a verifier scores from Llama 3.1, rounded to labels E to A. Most deepseek-r1 answers land on A, while about 18 to 28 minicpm5-2b answers per criterion land on B.">
  </div>
  <figcaption>Most deepseek-r1 answers land on A, while a clear group of minicpm5-2b answers lands on B.</figcaption>
</figure>

The mean verifier scores are 0.95 for deepseek-r1 against 0.90 to 0.93 for minicpm5-2b, so the Llama 3.1 verifier ranks deepseek-r1 ahead on every criterion.

### A different verifier flips the ranking

I repeated the verifier run with the Nimble verifier in place of the Llama 3.1 verifier:

<figure class="media-frame">
  <div class="media-frame-window">
    <div class="media-frame-bar"><span class="media-frame-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="media-frame-title">Verifier scores: Nimble</span></div>
    <img src="/assets/images/llm-as-a-verifier/nimble_reward_distribution.png" alt="Bar charts of LLM as a verifier scores from the Nimble verifier, rounded to labels E to A. Most minicpm5-2b answers land on A, while some deepseek-r1 answers drop to C and D.">
  </div>
  <figcaption>With the Nimble verifier, minicpm5-2b scores higher, and some deepseek-r1 answers drop to C and D.</figcaption>
</figure>

This time, minicpm5-2b comes out ahead on every criterion, with mean verifier scores of 0.96 to 0.97 against 0.90. Which answer model ranks first depends on which verifier is used.

Only one of these rankings is right. deepseek-r1 is the weaker answer model: at least 24 of its candidate answers are wrong, while minicpm5-2b gets at most 17 questions wrong (this follows from the 66 / 7 / 27 split shown in the last section). The judge and the Llama 3.1 verifier both rank the weaker answer model first. Only the Nimble verifier gets the order right.

### Which evaluator picks the correct candidate answer?

To test the evaluators directly, I looked for questions where exactly one answer model is correct. Only those questions have a single right choice:

| Outcome                       | Questions | Usable for picking the correct candidate answer? |
|-------------------------------|-----------|--------------------------------------------------|
| Both answer models correct    | 66        | No: both answers are correct                     |
| Both answer models wrong      | 7         | No: neither answer is correct                    |
| Only one answer model correct | 27        | Yes: there is exactly one correct pick           |

On each of the 27 questions:

1. One answer model's candidate answer is correct and the other's is wrong. The reference answer from GSM8K shows which, but the evaluators never see it.
2. Each evaluator scores both candidate answers for correctness.
3. The candidate answer it scores higher is its pick: the one it believes is correct.

The table shows how often each evaluator picked the correct candidate answer. Green marks the best evaluator:

<table class="data-table">
  <thead>
    <tr><th>Evaluator</th><th>Right</th><th>Wrong</th><th>Tie</th><th>Accuracy</th></tr>
  </thead>
  <tbody>
    <tr><td>Nimble verifier</td><td>21</td><td>6</td><td>0</td><td class="is-better">77.8%<span class="cell-note">21 / 27</span></td></tr>
    <tr><td>Llama 3.1 verifier</td><td>14</td><td>13</td><td>0</td><td>51.9%<span class="cell-note">14 / 27</span></td></tr>
    <tr><td>Llama 3.1 judge</td><td>9</td><td>0</td><td>18</td><td>33.3%<span class="cell-note">9 / 27</span></td></tr>
  </tbody>
</table>

The columns mean:

- **Right:** the evaluator scored the correct candidate answer higher.
- **Wrong:** the evaluator scored the wrong candidate answer higher, so it picked the answer model that made the mistake.
- **Tie:** the evaluator gave both candidate answers the same score, so it could not choose. A tie does not count as a correct pick.

The Nimble verifier picks the correct candidate answer most often: 21 times out of 27. The Llama 3.1 judge never picks the wrong candidate answer, but it ties on 18 of the 27 questions. The Llama 3.1 verifier breaks every tie, but it is right only 14 times, close to a coin flip.

---

## Conclusion

An LLM as a judge gives most candidate answers the same score, so it cannot tell answer models apart. An LLM as a verifier turns the probability of every label into a continuous score, but that score is only as reliable as the probabilities behind it.

On the 27 questions where exactly one answer model was correct, the judge tied on 18, and the Llama 3.1 verifier was right only 14 times. The Nimble verifier, a decision model that returns clean label probabilities, was right 21 times and was the only evaluator to rank the stronger answer model first. Each candidate answer was scored once, so these results do not yet account for single-run bias.
