# micro-scout

A research project for a fast, local code-context scout.

The scout is designed to find useful source snippets, account for relationships between symbols, and pass context to a larger model through a small agent harness: the loop that manages the model and its tools.

## Status

The research report and experiment plan are available. Implementation, trained weights, and project-specific benchmark results are not available yet.

## Documentation

- [Research and development plan](docs/RESEARCH.md): related work, Graphify, architecture, data, training, evaluation, resources, and an eight-week roadmap.
- Research date: September 16, 2026.
- The current budget excludes calls to the teacher and main models. It covers training the scout and the supporting infrastructure.

## Proposed architecture

```text
Repository and working-tree changes
    → symbol graph and search indexes
    → candidate retrieval
    → small model for selection and action choice
    → source snippets with verified locations
    → larger model and solution verification
```

Repository facts live in an external, updatable index. The model learns to select useful context and search actions for unfamiliar projects.

One proposed training setup uses GPT-5.6 Luna to generate examples for the local scout, then evaluates the scout with GPT-6 Astra as the main solver. The research report describes how to check whether the learned retrieval behavior transfers between them.

## Initial experiments

1. Build a minimal harness with search, symbol reading, and graph traversal.
2. Compare conventional search, graph search, and an existing reranker on the same tasks.
3. Measure task success, end-to-end latency, context size, and reference freshness.
4. Evaluate a custom encoder, then reduce its size.
5. If the benefit is confirmed, train action selection and search-budget allocation.

## Success criterion

Reduce time to solution while maintaining task success on unfamiliar repositories. Evaluation covers the full agent loop, additional reads, and index updates, as well as individual model-call latency.

Model sizes, latency targets, and budgets in the report are hypotheses to test. They are not measured micro-scout results.
