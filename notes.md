# EXERCISE

## The Problem: 

A department wants to test a new benefits calculator but cannot use real citizen data due to privacy.

## The Task: 

Use an LLM to generate 100 diverse "persona" profiles (income, family size, disability status) that are statistically plausible but entirely synthetic.

---

# INITIAL THOUGHTS

Is this only a prompt engineering problem? single node generator?
We also need a structured schema for the created profiles
Should have the LLM create them one at a time or in batches? 
one shot vs few shot?
How do we evaluate the realism and diversity of the dataset? 
Check the final distributions, should match national statistics? 
How do we check for bias => stratified analysis => and loop to add more / different profiles within specific groups?

- interative generation => generator node
- audit (diversity and bias) => grader node (deterministic)
- sefl-correcting loop based on audit result