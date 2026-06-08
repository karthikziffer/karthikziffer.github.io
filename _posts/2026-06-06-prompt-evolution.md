---
layout: post
title: "Prompt Evolution"
author: "Karthik"
categories: journal
tags: [agentic ai, prompt engineering]
---

## Introduction

This blog aims to develop an architectural strategy for storing prompts of agentic AI solutions in a graph database (considered Neo4j for this example). These prompts will be versioned over changes instead of mainatining the versioning in github, with metadata added to trace the version description. 

<br>
    
## Data Structure

The approach involves representing prompts as structured knowledge using nodes and entities in Neo4j. This representation decomposes information into clear, interconnected parts, making it easier to manage and evolve over time. Unlike traditional repositories, Neo4j provides a more efficient way to track changes, debug errors, and maintain security by enforcing strict data integrity and access controls.

Prompts are divided into nodes and entities based on their structure and relationships. For instance, a classification prompt might be represented as nodes for the task (e.g., "classification"), model type (e.g., "transformer"), and specific classes (e.g., "cat", "dog"). This structured format facilitates efficient querying and modification of prompts.

The data structure in Neo4j can be defined using labels (e.g., `Prompt`, `Task`, `ModelType`) and relationships (e.g., `HAS_TASK`, `USES_MODEL_TYPE`). Each prompt node would contain properties like its text, description, version, and metadata. Relationships between nodes represent the semantic connections between prompts and their components.

This approach is flexible and can be adapted to various AI and machine learning applications. Its dynamic evolution allows it to accommodate new types of prompts without requiring significant changes to the existing framework.

<br>
    
## Security

Storing prompts in Neo4j offers an advantage by making it easier to debug errors on each prompt node and entity. Another interesting application is that if nodes and entities are stored as embeddings, similarity can be calculated across these prompt nodes and entities using vector embeddings from the RAG vector store.

For the security, storing prompts in Neo4j avoids prompt injection attacks since the attacker cannot introduce any new corrupted prompts. This is because Neo4j enforces strict data integrity and authentication mechanisms. Additionally, the use of access controls ensures that only authorized users can modify or delete prompts, thereby preventing malicious modifications.

<br>
    
## Example

Prompts are decomposed into nodes and entities based on their structure and relationships. For example, a classification prompt might be broken down into nodes for the task (e.g., "classification"), the model type (e.g., "transformer"), and the specific classes (e.g., "cat", "dog"). This structured decomposition allows for efficient querying, retrieval, and modification of prompts.

The data structure for the prompts in Neo4j can be created using labels (e.g., `Prompt`, `Task`, `ModelType`) and relationships (e.g., `HAS_TASK`, `USES_MODEL_TYPE`). Each prompt node would contain properties such as its text, description, version, and metadata. Relationships between nodes represent the semantic connections between prompts and their components.

This project is not specific to a single use case but can be adapted for various applications involving AI and machine learning. Its dynamic evolution allows it to accommodate new types of prompts and data structures without requiring significant changes to the existing framework.

<br>
    
## Example Prompts


### Example 1

```md
Classify this image as either a cat or a dog
```



```
(node:Prompt {text: "Classify this image as either a cat or a dog.", version: "1.0"})
(node:Task {name: "Classification", description: "Determining the category of an object."})
(node:ModelType {name: "Transformer", description: "A type of neural network model."})

(relation:HAS_TASK {confidence: 0.95})
(relation:USES_MODEL_TYPE {accuracy: 0.98})
```

Example of storing them in Neo4j:

```
MATCH (p:Prompt), (t:Task), (m:ModelType)
WHERE p.text = "Classify this image as either a cat or a dog."
AND t.name = "Classification"
AND m.name = "Transformer"

CREATE (p)-[:HAS_TASK]->(t),
       (p)-[:USES_MODEL_TYPE]->(m)
```


<br>
    
### Example 2

We have seen a simple and easy prompt, but what happens to multi-line complex prompts? For example:

```md
Act as an IT Specialist/Expert/System Engineer. You are a seasoned professional in the IT domain. Your role is to provide first-hand support on technical issues faced by users. You will:
- Utilize your extensive knowledge in computer science, network infrastructure, and IT security to solve problems.
- Offer solutions in intelligent, simple, and understandable language for people of all levels.
- Explain solutions step by step with bullet points, using technical details when necessary.
- Address and resolve technical issues directly affecting users.
- Develop training programs focused on technical skills and customer interaction.
- Implement effective communication channels within the team.
- Foster a collaborative and supportive team environment.
- Design escalation and resolution processes for complex customer issues.
- Monitor team performance and provide constructive feedback.

Rules:
- Prioritize customer satisfaction.
- Ensure clarity and simplicity in explanations.

Your first task is to solve the problem: "my laptop gets an error with a blue screen."
```



The complexity increases when a multi line prompts need to be decomposed into Neo4j representation.

We create a general pattern to use for multi line prompts. The original prompt is broken down into Prompt entity, role or task entities, responsibilites or action entities, rules and constraints. 



![](/assets/images/prompt-evolution-framework.png)


<br>
    
1. **Prompt Entity**: Represents the overall prompt text.
2. **Role/Task Entities**: Break down the tasks mentioned in the prompt into discrete roles or tasks.
3. **Responsibilities/Actions Entities**: Identify the actions associated with each role/task.
4. **Rules and Constraints**: Capture any rules or constraints specified in the prompt.


<br>
    

#### Prompt Entity

```plaintext
(node:Prompt {text: "Act as an IT Specialist/Expert/System Engineer...", version: "1.0"})
```


<br>
    
#### Role/Task Entities

- **Role**: IT Specialist/Expert/System Engineer
  - (node:Role {name: "IT Specialist", description: "A seasoned professional in the IT domain."})
  - (node:Role {name: "System Engineer", description: "A person who designs and maintains computer systems."})

```plaintext
(node:Task {name: "Provide first-hand support on technical issues"})
(node:Task {name: "Offer solutions in intelligent, simple, and understandable language"})
(node:Task {name: "Explain solutions step by step with bullet points"})
(node:Task {name: "Address and resolve technical issues directly affecting users"})
(node:Task {name: "Develop training programs focused on technical skills and customer interaction"})
(node:Task {name: "Implement effective communication channels within the team"})
(node:Task {name: "Foster a collaborative and supportive team environment"})
(node:Task {name: "Design escalation and resolution processes for complex customer issues"})
(node:Task {name: "Monitor team performance and provide constructive feedback"})
```

<br>
    
#### Responsibilities/Actions Entities

- **Responsibilities**: Utilize extensive knowledge, offer simple solutions, step-by-step explanations.
  - (node:Responsibility {text: "Utilize extensive knowledge in computer science, network infrastructure, and IT security."})
  - (node:Responsibility {text: "Offer solutions in intelligent, simple, and understandable language for people of all levels."})
  - (node:Responsibility {text: "Explain solutions step by step with bullet points, using technical details when necessary."})

- **Actions**: Develop training programs, implement communication channels, foster a team environment.
  - (node:Action {text: "Develop training programs focused on technical skills and customer interaction."})
  - (node:Action {text: "Implement effective communication channels within the team."})
  - (node:Action {text: "Foster a collaborative and supportive team environment."})


<br>


#### Rules/Constraints Entities

- **Rule**: Prioritize customer satisfaction.
- **Constraint**: Ensure clarity and simplicity in explanations.

```plaintext
(node:Rule {text: "Prioritize customer satisfaction"})
(node:Constraint {text: "Ensure clarity and simplicity in explanations"})
```

<br>


#### Relationships

1. **Prompt to Role**
2. **Role to Task**
3. **Task to Responsibility/Action**
4. **Prompt to Rule/Constraint**


<br>


```plaintext
(relation:HAS_ROLE {confidence: 0.95})
(relation:PERFORMS_TASK {confidence: 0.90})
(relation:HAS_RESPONSIBILITY {confidence: 0.85})
(relation:FULFILLS_CONSTRAINT {confidence: 0.92})

(match (p:Prompt), (r:Role)
WHERE p.text contains r.description
CREATE (p)-[:HAS_ROLE]->(r))

(match (t:Task), (res:Responsibility)
WHERE t.name = "Provide first-hand support on technical issues" AND res.text contains "Utilize extensive knowledge"
CREATE (t)-[:PERFORMS_TASK]->(res))

(match (t:Task), (act:Action)
WHERE t.name = "Develop training programs focused on technical skills and customer interaction" AND act.text contains "Develop training programs"
CREATE (t)-[:PERFORMS_TASK]->(act))

(match (p:Prompt), (rule:Rule)
WHERE p.text contains rule.text
CREATE (p)-[:FULFILLS_CONSTRAINT]->(rule))
```

<br>
    

## Evolution

The version of the prompt is done in Neo4j using nodes for different version of prompts and relationships to track their history. Each version contains a timestamp indicating when it was created, and links back to the prompt's data. When updating a prompt, you create a new version node that includes the updated content and links to both the old and new versions. This allows querying the database as if it were at any given point in time, capturing historical changes accurately. 


<br>


## Results

    
A sample size of [1828 prompts](https://raw.githubusercontent.com/f/prompts.chat/main/PROMPTS.md) is taken for the evaluation of the framework to determine if these prompts can be represented within this decomposed role or task entities, responsibility or actions entities, and rules and constraints. 


First result shows on the sample prompts, what percentage has their roles, tasks, responsibilities, actions, and rules or constraints.


![](/assets/images/clean_threshold_comparison.png)



<br>


Second result is to check what is the distribution of the prompts into proposed framework components. 

The image shows histograms representing the frequency of prompts categorized by their roles, tasks, responsibilities, actions, and rules or constraints. The plots provide a visual overview of how well the current framework can represent various types of prompts.

![](/assets/images/all_metrics_histograms.png)


<br>


## Conclusion

This blog has outlined an architectural strategy for storing prompts of agentic AI solutions in a graph database, specifically Neo4j. The approach involves representing prompts as structured knowledge using nodes and entities, which allows for efficient management and evolution over time. By decomposing information into clear, interconnected parts, we can handle versioning, debugging, and security more effectively.

The example provided demonstrates how complex multi-line prompts can be broken down into individual roles, tasks, responsibilities, actions, and rules. This structured decomposition enables precise querying and modification of prompts, making it easier to adapt the framework to various AI and machine learning applications.

The results from the sample evaluation further support the effectiveness of this approach in handling a diverse range of prompts. The histograms illustrate how well the current framework can represent different types of tasks and responsibilities, indicating a promising direction for future work.
