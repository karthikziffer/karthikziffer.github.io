---
layout: post
title: "Hallucination detection without LLM"
author: "Karthik"
categories: journal
tags: [agentic ai, prompt engineering]
---

Hallucination detection works at two levels: semantic and factual. Which one you need depends on the document. A software version number rarely contradicts itself, so a factual check is enough there. A support log is trickier: a listed 25% discount might only apply in summer, and catching that kind of contradiction takes semantic understanding, not just fact-checking. Most real systems need both. This post covers a non-LLM factual technique: breaking text down into facts and checking them against the source, instead of asking a model to judge its own answer.


---

## The Idea

The idea is simple: break text into facts with LLM. Once a passage is broken into who did what, and what we know about them, that structure turns out to be useful for more than spotting hallucinations. There are three applications at the end of this post. But first, here's how the decomposition itself works, using hallucination detection as the running example.

This only works under a few assumptions:

- **Spelling and grammar must be accurate.** Nothing here corrects errors before parsing.
- **The text must read as generic English prose.** The parser and tagger are trained on general-purpose corpora, so jargon, code snippets, and unusual phrasing degrade their output.
- **Entities must be the kind of thing spaCy's NER model would recognize.** Its knowledge is frozen at training time, so a very recent or highly niche named entity may be missed or mislabeled.


---

## Meet the Text

One example runs through this whole post, so a single fact can be followed from sentence to final answer.

**Context text:**

<div markdown="0" style="font-size:0.85rem; margin-bottom:0.75rem;">
  <span style="background-color:rgba(255,222,3,0.4); padding:2px 8px; border-radius:4px; margin-right:6px; display:inline-block; margin-bottom:4px;">Date</span>
  <span style="background-color:rgba(3,168,124,0.22); padding:2px 8px; border-radius:4px; margin-right:6px; display:inline-block; margin-bottom:4px;">Place</span>
  <span style="background-color:rgba(120,50,226,0.2); padding:2px 8px; border-radius:4px; margin-right:6px; display:inline-block; margin-bottom:4px;">Nationality / group</span>
  <span style="background-color:rgba(53,189,255,0.28); padding:2px 8px; border-radius:4px; display:inline-block; margin-bottom:4px;">Organization</span>
</div>

<blockquote markdown="0">
"Arthur's Magazine (<span style="background-color:rgba(255,222,3,0.4); border-radius:3px; padding:0 2px;">1844-1846</span>) was an <span style="background-color:rgba(120,50,226,0.2); border-radius:3px; padding:0 2px;">American</span> literary periodical published in <span style="background-color:rgba(3,168,124,0.22); border-radius:3px; padding:0 2px;">Philadelphia</span> in <span style="background-color:rgba(255,222,3,0.4); border-radius:3px; padding:0 2px;">the 19th century</span>. First for Women is a woman's magazine published by <span style="background-color:rgba(53,189,255,0.28); border-radius:3px; padding:0 2px;">Bauer Media Group</span> in the <span style="background-color:rgba(3,168,124,0.22); border-radius:3px; padding:0 2px;">USA</span>."
</blockquote>

**Question:**

> "Which magazine was started first, Arthur's Magazine or First for Women?"

Everything below starts with just the context text.


---

## Step 1: Split Into Sentences

The first job is cutting the context text into individual sentences. That sounds like splitting at every period, but periods also show up inside dates, abbreviations, and titles. A naive split would mangle "1844-1846" and "the 19th century."

This is done with spaCy, a library built for natural language processing. Under the hood, splitting is a side effect of spaCy's **parser**, the component that detects how words relate to each other grammatically.

There are two types of parser. A **constituency parser** breaks a sentence into nested phrases, like a tree. A **dependency parser**, the one used here, instead draws a direct link between every pair of words where one depends on the other: subject to verb, verb to object, adjective to the noun it modifies.

Because splitting falls out of this grammatical model rather than a punctuation rule, entity names and dates stay intact instead of getting cut on their internal periods and hyphens.

> **Example:** a naive period-split would cut "Arthur's Magazine (1844-1846)" into two fragments, "Arthur's Magazine (1844" and "1846)". spaCy's parser-driven split keeps the whole sentence intact.

![](/assets/images/fact-decompose-step1.svg)


---

## Step 2: Find the Facts

With the text split into sentences, this step finds every verb, then sorts the words around it into buckets: whoever or whatever does the action is the subject; whatever receives it is the object; any "in," "by," or "at" detail goes into a separate list.

If the parser can't find this structure, say when the information before the verb is unusually complicated, NER (Named Entity Recognition) is used instead, extracting tags like `DATE`, `ORG`, and others.

This subject-predicate-object structure has a name: a **triple**. This step's output merges the parser's triple with whatever NER tags land inside it. The parse gives the connection, subject did X to object; NER gives the values, that object is a place, that subject is a person.

> **Example:** from "Arthur's Magazine (1844-1846) was ... published in Philadelphia in the 19th century," the triple that comes out looks like:
> ```
> subject: Arthur's Magazine (1844-1846)
> predicate: publish
> attributes: in Philadelphia (Place), in the 19th century (Date)
> ```

![](/assets/images/fact-decompose-step2.svg)

### How NER Works

spaCy's NER is a statistical model trained on human-annotated data. It makes a **per-token prediction**, not a dictionary lookup, since a dictionary would need constant updates to keep up with new names.

A **token** is a single unit of text after splitting; punctuation gets its own token. Each token gets one of five labels: **Outside** (not part of an entity), **Begin** (start of one), **In** (middle of one), **Last** (end of one), or **Unit** (a whole one-token entity by itself).

| Tag | Meaning | Example (entity: "Bauer Media Group" → ORG) |
|---|---|---|
| **B** — Begin | first token of a multi-token entity | Bauer → B-ORG |
| **I** — In | middle token(s) of a multi-token entity | Media → I-ORG |
| **L** — Last | final token of a multi-token entity | Group → L-ORG |
| **O** — Outside | not part of any entity | published, by, in → O |
| **U** — Unit | a single-token entity, whole by itself | Philadelphia → U-GPE |

**Example:** tagging a full sentence from the demo text shows how this plays out:

> "First for Women is published by Bauer Media Group in Philadelphia."

```
First      O
for        O
Women      O
is         O
published  O
by         O
Bauer      B-ORG
Media      I-ORG
Group      L-ORG
in         O
Philadelphia  U-GPE
.          O
```

This is called the **BILOU scheme**. Stringing consecutive B/I/L tags together recovers a multi-word span, like "the 19th century," as one `DATE`.

Worth distinguishing a **tagger** from a **parser** here: the parser figures out how words connect; the tagger figures out each word's part of speech. Both share the same word representations, `tok2vec`, and weigh three signals per decision: the word's own embedding, learned from training data so it generalizes to unseen words, unlike a plain lookup table; its shape, capitalized or not, digits or not; and its surrounding context.

The outcome of this step is a **fact** per sentence: a subject, an action, and whatever other details go with it. "Fact" and "triple" name the same thing: triple describes its shape, three parts; fact describes what it means.


---

## Step 3: Build a Fact Table

Facts from every sentence get combined by subject, so scattered facts about the same entity land in one place. This merged structure is a **fact table**: entity name mapped to every typed attribute (`DATE`, `GPE`, `ORG`, `NORP`, etc.) ever attached to it, deduplicated.

**Example fact table** for the demo text:

```json
{
  "Arthur's Magazine (1844-1846)": {
    "NORP": ["American"],
    "GPE": ["Philadelphia"],
    "DATE": ["the 19th century"]
  },
  "First for Women": {
    "ORG": ["Bauer Media Group"],
    "GPE": ["USA"]
  }
}
```

Notice "First for Women" has no `DATE` entry: the context text never states when it started. Keep that in mind, it's the whole point by the end.

![](/assets/images/fact-decompose-step3.svg)


---

## Step 4: Answer From the Fact Table

This step takes the question, creates facts from it too, and answers by looking things up in the fact table.

### a. Break the question apart

Figure out which two things are being compared.

**Example:** for *"Which magazine was started first, Arthur's Magazine or First for Women?"*:

```
candidates: ["Arthur's Magazine", "First for Women"]
attribute needed: DATE
direction: min (earliest)
```

### b. Link the candidates to the fact table

Using fuzzy logic matching. The question's wording won't match the fact table's key exactly, so this step scores similarity rather than doing an exact lookup.

> **Example:**
> ```
> "Arthur's Magazine"  →  "Arthur's Magazine (1844-1846)"   (72% match)
> "First for Women"     →  "First for Women"                 (100% match)
> ```

### c. Get the requested attribute's value

For each candidate, from the fact table.

### d. Compare the values

Return the answer, or if it's missing, return "not answerable" instead of guessing.

> **Example:**
> ```
> "First for Women" has no DATE in the fact table.
> answer: not answerable — couldn't find a date for First for Women.
> ```

![](/assets/images/fact-decompose-step4.svg)

These four steps make up the pipeline this post is walking through. This is the payoff for hallucination detection: if the answer is missing here, but an LLM asked the same question still returns a confident answer anyway, that model is probably hallucinating.


---

## Applications

Keeping the clean-text assumption from earlier in mind, there are three applications worth mentioning.

### 1. A pre-step for RAG or agentic workflows, before context retrieval

A RAG system at millions-of-documents scale breaks under peak traffic if every chunk lives in one index, so the standard fix is sharding across vector databases. Finding each chunk's intent and entity with a non-LLM technique, instead of an LLM, lets you hash chunks to a shard and search only that shard. Smaller search space, faster retrieval.

> **Example:** a query can be filtered to search only document chunks tagged with the entity "Philadelphia," instead of scanning the entire index.

Similar in spirit to GraphRAG, which pairs a graph of intent/entity relationships with vector embeddings. But running both inside one system bottlenecks at scale, so a distributed, layered search works better.

### 2. Hallucination detection

As walked through above: once both context text and question are fact-decomposed, their intents and entities can be checked against each other. A missing relationship means a missing answer, which raises the probability of hallucination.

### 3. Evaluation test-case generation

The same fact table that answers questions can also generate high-quality evaluation datasets automatically, to test other systems.


---

## Metrics to Evaluate

Taking the hallucination application to detect the correct answer for the evaluation task. It is measured as a match rate, run the pipeline over a labeled evaluation dataset, a set of (context text, question, expected answer) triples, and check what percentage return the expected answer.

This needs a **Context-Based QA** dataset specifically: each example already pairs a question with the passage its answer comes from, unlike open-domain QA with no fixed source. The running example (Arthur's Magazine vs. First for Women) is drawn from one such dataset, HotpotQA; SQuAD is another.

**Example evaluation samples**, taken from SQuAD:

| Sample | Context | Question | Expected Answer |
|---|---|---|---|
| 1 | "The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). ..." | "Where is the headquarters of the Congregation of the Holy Cross?" | `Rome` |
| 2 | "All of Notre Dame's undergraduate students are a part of one of the five undergraduate colleges at the school or are in the First Year of Studies program. ..." | "How many colleges for undergraduates are at Notre Dame?" | `five` |

Sample 1 needs a `GPE` attribute (a place), sample 2 needs a `CARDINAL` attribute (a number), so together they already exercise two different branches of the attribute-type lookup from Step 4.


---

## Evaluation Results

Running this metric on 58 single-hop questions sampled from SQuAD:

![](/assets/images/hallucination/02_total_vs_matched.png)

Only 17 of 58 questions were answered correctly, a 29% match rate.

Breaking down the 41 failures:

![](/assets/images/hallucination/03_failure_reasons.png)

76% of the failures, 31 of 41, are "answered incorrectly": the pipeline committed to a wrong value instead of admitting defeat. The remaining 10 are "attribute missing on entity": the entity linked correctly, but the needed attribute type (say, `DATE`) was never attached to it, so the pipeline answered "not answerable" instead.


---

## Why the Match Rate Is Low

Approximately 30% is a good percentage for a complex evaluation dataset, and further improvements could bring better results. SQuAD contexts run several sentences deep, with clauses nested inside clauses, exactly where the parser and tagger from Step 2 struggle to hold onto the right subject and attribute. That's what the failure-reasons chart is really showing: "answered incorrectly" is rarely a missing fact. It's the parser attaching an attribute to the wrong entity, or NER tagging the wrong span, in a sentence too long or complex to track cleanly.

There are two ways to push past this, one LLM-based and one not.

### Replace the parser and tagger with an LLM

An LLM holds onto subject-object relationships across long, complicated sentences better than spaCy's dependency parser and tagger. Swap those two components for an LLM call, keep the rest of the pipeline (fact table, fuzzy linking, "not answerable" logic) as-is, and the match rate on harder contexts should rise directly.

### Or, staying non-LLM: add coreference resolution

Long contexts refer back to the same entity with a pronoun or a shortened name a sentence later ("Notre Dame ... the university ... it ..."), and the pipeline currently treats each as an unrelated subject, since fact-table grouping is exact-string matching (Step 3). Coreference resolution would link those mentions to one entity before the fact table gets built, so facts scattered across a long context merge under one subject instead of splintering. Fully non-LLM, and it targets the multi-hop failure mode directly rather than parsing accuracy in general.


---

## Closing Thought

This isn't a replacement for an LLM. It's a way to build a structured understanding of your data without one. The best practice is to treat text as sentences, facts, not just a blob of words.
