# Morphological Analysis Task

The response keeps the existing parallel arrays (`tokens`, `pos_tags`, `lemmas`,
`morphological_features`). Validation rejects any response where their lengths
differ. Index `i` in each array refers to the same token; all four arrays may
be empty for empty input.

::: openaivec.task.nlp.morphological_analysis
