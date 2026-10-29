# Blog writing instructions

Guidelines for writing technical blog posts on this site.

The goal: a reader who knows a little about the topic can follow every sentence on the first read.

---

## 1. Consistent vocabulary

Pick one name for each thing and use that exact name for the whole post.

**Before writing, list your terms.** Put a short term list at the top of your draft (delete it before publishing):

| Thing                | Name to use        | Don't switch to              |
|----------------------|--------------------|------------------------------|
| The thing you built  | `<one fixed name>` | `<synonyms you might drift to>` |
| Each main component  | `<one fixed name>` | `<synonyms>`                 |
| Each external service | `<official name>` | `<shortened or informal names>` |

Rules:

- **Define a term once, on first use, in plain words.** Put the definition in parentheses or right after a colon:
  - "a webhook (a URL another service calls when something happens)"
  - "a race condition: two tasks changing the same data at the same time"
- **After defining a term, never swap in a synonym.** A new word makes the reader think it's a new thing. If the intro says "worker", every later section says "worker", not "job runner" or "process".
- **Keep the same capitalization every time.** Use the official spelling of product and company names, and use it every time.
- **Pair terms the same way.** If one section compares "A vs B", every section uses "A" and "B" with the same wording.
- **Use code formatting for things a reader would type or see in code:** function names, file names, commands, config keys, API fields. Format the name the same way every time it appears.
- **Use bold for UI labels exactly as they appear on screen.**

## 2. Easily understood, everyday vocabulary

Write with words the reader already uses. If a simpler word works, use it.

| Instead of        | Write          |
|-------------------|----------------|
| utilize           | use            |
| leverage          | use            |
| facilitate        | help, let      |
| in order to       | to             |
| subsequently      | then, later    |
| prior to          | before         |
| commence          | start          |
| terminate         | stop, end      |
| sufficient        | enough         |
| robust, seamless  | (cut it, or say what it actually does) |

Sentence rules:

- **One idea per sentence.** If a sentence has "and" joining two separate ideas, split it.
- **Keep most sentences under 25 words.**
- **Use active voice, present tense.** "The server sends the response", not "The response is sent by the server".
- **Every sentence needs a subject.** "It gives less visibility", not "Gives less visibility".
- **Use articles** ("a", "an", "the"). "The client sends a request", not "client sends request".
- **Write "I" for what you built and did.** Write "you" only for instructions the reader follows.
- **Name who does what.** When several parts are involved, name the part doing each step. Readers get lost when "it" could mean more than one thing.
- **Say the limitations plainly.** Saying what a solution doesn't do builds trust.
- **Explain jargon by what it does, not just its name.** Follow a technical term with what it means for the reader: "That means ..."

## 3. Post structure

Follow this order:

1. **Front matter**
   ```yaml
   ---
   layout: post
   title: "Short descriptive title"   # sentence case, no clickbait
   author: "Karthik"
   categories: journal
   tags: [topic-one, topic-two]       # lowercase, hyphenated
   ---
   ```
2. **The problem (1 short paragraph).** What goes wrong today and why it matters.
3. **What I built or found (1 sentence + a bullet list).** Each bullet starts with a bold summary sentence, then explains:
   ```markdown
   - **Does the first thing.** One or two sentences on how.
   - **Does the second thing.** One or two sentences on how.
   ```
4. **Limitations (1 short paragraph).** What the solution doesn't do.
5. **"In this post:" as a numbered list.** Each item matches an `##` heading below, in the same order and with the same wording.
6. **Sections**, each separated by `---`. A common order:
   - Background: how the existing system works
   - Key concept: the idea the solution depends on
   - How it works
   - How to set it up or reproduce it
7. **Conclusion (1 or 2 short paragraphs).** Restate the problem in one sentence, say what the solution does, and repeat the main limitation.

Heading rules:

- `##` for main sections, `###` for subsections. Don't skip levels, and don't use `#` in the body (the title is already `#`).
- Use sentence case: "How the cache works", not "How The Cache Works".
- Make headings describe the content: "Why the order matters", "When to use which".
- For a numbered series, use the same pattern for every heading and add the class:
  ```markdown
  ### Step 1: Short name
  {: .layer-heading}
  ```

## 4. Content elements

Use each element for one job only.

| Element          | Use it for                                     | Pattern |
|------------------|------------------------------------------------|---------|
| Paragraph        | Explaining one idea                            | 1 to 4 sentences |
| Bullet list      | Items without an order                         | Bold lead, then the detail: `- **Term:** explanation` |
| Numbered list    | Steps in order                                 | Start each step with a verb: "Install", "Edit", "Restart" |
| Table            | Comparing things on the same points            | Things as columns, points as rows |
| Code block       | Anything the reader types or reads as code     | Always set the language: ` ```json `, ` ```bash `, ` ```python ` |
| Diagram          | A flow, sequence, or structure                 | `figure.diagram`, see section 6 |
| GIF / screenshot | Showing the result in action                   | `figure.media-frame`, see section 6 |

Rules:

- **Introduce every code block, diagram, and table with a sentence ending in a colon.** "The config file looks like this:"
- **Follow every code block with one sentence saying what it means or what happens next.**
- **Keep list items parallel.** All bullets in a list start with the same part of speech (all verbs, or all nouns).
- **End bullets' bold leads with a colon or a period, and use the same one throughout the list.**
- **Show real examples** (actual input, output, or config), not pseudo-code.

## 5. Font design (already set by the theme)

The theme in `assets/css/custom.css` sets the fonts. Don't add inline styles, `<font>` tags, or new font imports to a post.

| Role             | Font               | Size / weight              |
|------------------|--------------------|----------------------------|
| Post title       | Newsreader (serif) | 3rem, weight 500           |
| `##` headings    | Newsreader (serif) | 2.1rem, weight 500         |
| `###` headings   | IBM Plex Sans      | 1.25rem, weight 600        |
| `####` headings  | IBM Plex Sans      | 0.85rem, weight 600, muted |
| Body text        | IBM Plex Sans      | 1.1rem, line height 1.75   |
| Code             | IBM Plex Mono      | -                          |
| Captions         | IBM Plex Sans      | 0.85rem, italic, muted     |
| Diagram labels   | IBM Plex Sans      | 13px bold / 11px small     |

Colors come from the theme tokens. Use these, never raw hex values:

- `--ed-ink`: main text
- `--ed-ink-muted`: secondary text, arrows, captions
- `--ed-accent`: links, and the one highlighted box in a diagram
- `--ed-rule`: dividers, dashed boxes
- `--ed-code-bg` / `--ed-code-border`: normal diagram boxes
- `--ed-good-bg` / `--ed-good-ink`: table cells marking a better result. Write the table in HTML and add `class="is-better"` to the `<td>`, and say in the intro sentence what green means.

Use emphasis sparingly:

- **Bold** for bullet leads, UI labels, and at most one key phrase per paragraph.
- *Italic* rarely in body text.
- No ALL CAPS, no exclamation marks, no emoji.

## 6. Clean alignment

### Text column

- The body is a single 700px column. Diagrams and images fit inside it. Don't add wide tables or full-width HTML.
- Spacing is set by the theme (`--ed-flow` between paragraphs and lists, `--ed-block` around figures). Don't add `<br>` or empty lines for extra space.
- Use `---` only between `##` sections.

### Tables

Align the pipes in the Markdown source so the table is readable as plain text:

```markdown
|               | Option A          | Option B          |
|---------------|-------------------|-------------------|
| Point one     | Short phrase      | Short phrase      |
| Point two     | Short phrase      | Short phrase      |
```

Keep cells short (a phrase, not a sentence). Start every cell with a capital letter.

### Diagrams (inline SVG)

Use this template:

```html
<figure class="diagram">
<svg viewBox="0 0 700 112" role="img" aria-label="One sentence describing the whole diagram" class="diagram">
  <defs>
    <marker id="ah-UNIQUE" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"></path>
    </marker>
  </defs>

  <rect x="0" y="30" width="145" height="50" rx="3" class="box"></rect>
  <text x="72.5" y="52" text-anchor="middle" class="label-strong">Box title</text>
  <text x="72.5" y="68" text-anchor="middle" class="label-small">what it does</text>
  <!-- more boxes -->
  <path d="M145,55 L185,55" class="arrow" marker-end="url(#ah-UNIQUE)"></path>
</svg>
<figcaption>One sentence stating the takeaway.</figcaption>
</figure>
```

Alignment rules:

- **Width is always 700** (`viewBox="0 0 700 H"`), matching the text column. Only the height changes.
- **Use a grid.** Boxes in a row share the same `y`, `width`, and `height`, and the gaps between them are equal.
  - 4 boxes: width 145, gap 40 (x = 0, 185, 370, 555)
  - 3 boxes: width 180, gap 80 (x = 0, 260, 520)
  - 2 boxes: width 260, gap 180 (x = 0, 440)
- **Center text in its box.** Use `text-anchor="middle"` and `x = box x + width / 2`. The title line sits at `box y + 22`, the subtitle at `box y + 38`.
- **Arrows connect box edges at the vertical middle** (`y = box y + height / 2`). Arrow labels sit centered above the gap.
- **Use only the theme classes:** `box`, `box-accent`, `box-dashed`, `arrow`, `arrow-head`, `label`, `label-strong`, `label-small`, `label-muted`, `label-accent`, `lane-title`, `lane-rule`. No `fill=` or `stroke=` attributes.
- **Highlight at most one box** with `box-accent`: the part the post is about.
- **Give each marker a unique id** so diagrams on the same page don't clash.
- **Use the same terms in the diagram as in the text.**

### GIFs and screenshots

```html
<figure class="media-frame">
  <div class="media-frame-window">
    <div class="media-frame-bar"><span class="media-frame-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="media-frame-title">Short title</span></div>
    <img src="/assets/images/POST-FOLDER/file.gif" alt="What the image shows, in one sentence">
  </div>
  <figcaption>One sentence stating what to notice.</figcaption>
</figure>
```

- Store images in `/assets/images/<post-folder>/`.
- The window title is 1 to 3 words in sentence case.
- Alt text describes what is on screen. The caption says what to notice.

### Captions

Every figure has a caption. A caption is one or two short sentences stating the takeaway, not a label:

- Good: "The cache answers repeat requests, so the database only sees the first one."
- Weak: "Figure 1: Caching flow."

## 7. Checklist before publishing

- [ ] Each thing has one name, used the same way everywhere (check the term list).
- [ ] Every term is defined in plain words on first use.
- [ ] Product names are spelled and capitalized correctly every time.
- [ ] No sentence is missing its subject or articles.
- [ ] No sentence has more than one main idea or more than about 25 words.
- [ ] The "In this post" list matches the `##` headings exactly.
- [ ] Headings are in sentence case, and `##` sections are separated by `---`.
- [ ] Every code block has a language and an intro sentence.
- [ ] Every diagram is 700 wide, on a grid, uses only theme classes, and has an `aria-label` and a caption.
- [ ] Every image has alt text and a caption.
- [ ] No inline styles, raw colors, or extra fonts.
- [ ] Limitations are stated in the intro and the conclusion.
- [ ] Preview locally with `bundle exec jekyll serve` and check the post at phone width.
