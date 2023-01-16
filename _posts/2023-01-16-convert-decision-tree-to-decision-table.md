---
layout: post
title: "Convert decision tree to decision table"
author: "Karthik"
categories: journal
tags: [documentation,sample]







---



<br>

I was working on a project, where I wanted to analyze the decisions from the decision tree. I thought, converting a decision tree to decision table helps in the interpretation and analysis. There was just one stack-overflow solution for converting a decision tree to decision table, but it failed for my use case. Hence I wrote a parsing code to convert the decision tree to decision table. 

<br>

This blog post is for the conversion code. 

<br>

#### Input

```

|--- feature_2 <= 2.50
|   |--- feature_1 <= 2.00
|   |   |--- class: 1
|   |--- feature_1 >  2.00
|   |   |--- feature_0 <= 2.50
|   |   |   |--- class: 1
|   |   |--- feature_0 >  2.50
|   |   |   |--- class: 0
|--- feature_2 >  2.50
|   |--- feature_1 <= 1.00
|   |   |--- class: 0
|   |--- feature_1 >  1.00
|   |   |--- feature_0 <= 1.50
|   |   |   |--- class: 1
|   |   |--- feature_0 >  1.50
|   |   |   |--- class: 0

```



<br>



#### Code



```
import re

sequence = []
rules = []
previous_bar_count = 0

# split the decision tree text using line break 
tree_text_data = text_representation.split('\n')
"""
tree_text_data:

['|--- feature_2 <= 2.50',
 '|   |--- feature_1 <= 2.00',
 '|   |   |--- class: 1',
 '|   |--- feature_1 >  2.00',
 '|   |   |--- feature_0 <= 2.50',
 '|   |   |   |--- class: 1',
 '|   |   |--- feature_0 >  2.50',
 '|   |   |   |--- class: 0',
 '|--- feature_2 >  2.50',
 '|   |--- feature_1 <= 1.00',
 '|   |   |--- class: 0',
 '|   |--- feature_1 >  1.00',
 '|   |   |--- feature_0 <= 1.50',
 '|   |   |   |--- class: 1',
 '|   |   |--- feature_0 >  1.50',
 '|   |   |   |--- class: 0',
 '']
 """


for idx, _ in enumerate(tree_text_data):

  # count the pipe symbol, the count of pipe symbol in each line gives the tree depth information (example: parent, child1 etc)
  bar_count = _.count('|')

  # regex for extracting decision rule
  feature_rule = re.search("[a-z 0-9_.<=>:]+", _.replace(' ', ''))

  # condition for increasing pipe count (ex: 1, 2, 3, 4 ...)
  if (previous_bar_count+1 == bar_count) and (bar_count > 0) :
    sequence.append(feature_rule.group(0))

  # condition for pipe count break (ex: 1, 2, 3, 2)
  elif (previous_bar_count+1 != bar_count) and (bar_count > 0):
    rules.append(sequence)

    # update sequence with current iteration decision
    sequence = sequence[:bar_count-1]
    sequence.append(feature_rule.group(0))
    
  # if it's last condition, add it to rules list
  if idx+1 == len(tree_text_data):
        rules.append(sequence)
    
    
  # previous line count
  previous_bar_count = bar_count


```



<br>



#### Output

```
[
  [
    'feature_2<=2.50',
    'feature_1<=2.00',
    'class:1'
  ],
  [
    'feature_2<=2.50',
    'feature_1>2.00',
    'feature_0<=2.50',
    'class:1'
  ],
  [
    'feature_2<=2.50',
    'feature_1>2.00',
    'feature_0>2.50',
    'class:0'
  ],
  [
    'feature_2>2.50',
    'feature_1<=1.00',
    'class:0'
  ],
  [
    'feature_2>2.50',
    'feature_1>1.00',
    'feature_0<=1.50',
    'class:1'
  ],
  [
    'feature_2>2.50',
    'feature_1>1.00',
    'feature_0>1.50',
    'class:0'
  ]
]
```



<br>

#### Table

```
import pandas as pd

pd.DataFrame(rules)


```

| index | 0                  | 1                  | 2                  | 3       |
| ----- | ------------------ | ------------------ | ------------------ | ------- |
| 0     | feature\_2\<=2\.50 | feature\_2\<=0\.50 | feature\_3\<=1\.50 | class:1 |
| 1     | feature\_2\<=2\.50 | feature\_2\<=0\.50 | feature\_3\>1\.50  | class:0 |
| 2     | feature\_2\<=2\.50 | feature\_2\>0\.50  | class:1            |         |
| 3     | feature\_2\>2\.50  | feature\_1\<=1\.00 | class:0            |         |
| 4     | feature\_2\>2\.50  | feature\_1\>1\.00  | feature\_0\<=1\.50 | class:1 |
| 5     | feature\_2\>2\.50  | feature\_1\>1\.00  | feature\_0\>1\.50  | class:0 |



<br>

Feel free to use it in your project. 

