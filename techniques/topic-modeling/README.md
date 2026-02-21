# Topic Modeling

## What it is

Topic modeling is an unsupervised technique that discovers hidden thematic structures (called "topics") within a collection of text documents. Rather than reading thousands of documents yourself, you let an algorithm scan word co-occurrence patterns and group words that tend to appear together into coherent themes. The two workhorses of topic modeling are **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)**. Both take a collection of documents as input and produce two things: (1) a set of topics, each defined by a ranked list of words, and (2) a distribution over those topics for every document.

## Why it matters

Insurance and actuarial work generates enormous volumes of unstructured text -- claims adjuster notes, customer complaint logs, underwriting narratives, policy endorsement descriptions, medical reports, and survey responses. Manually categorizing this text is expensive and inconsistent. Topic modeling lets you:

- **Discover claim themes at scale**: Find out what types of incidents, injuries, or disputes keep showing up in claims notes without pre-defining categories.
- **Monitor customer sentiment**: Detect emerging complaint categories from call-center transcripts or NPS verbatims before they become systemic issues.
- **Enrich structured models**: Use the topic distribution of a document as a feature in a downstream classification or regression model (e.g., adding "claims note topic" as a predictor in a severity model).
- **Audit and compliance**: Quickly scan large corpora of policy language or regulatory filings for specific thematic content.

For an actuarial practitioner, this bridges the gap between "we have the data" and "we can actually use it" for text fields that typically sit ignored in data warehouses.

## When to use it

- You have a large collection of text documents (hundreds to tens of thousands) and want to discover what they talk about without labeling them first.
- Claims adjusters write free-text notes and you want to cluster them into incident types (e.g., water damage, slip-and-fall, vehicle collision).
- You want to track how customer feedback topics shift over time (topic trend analysis).
- You need to generate features from text data for a predictive model.
- You have policy description text and want to identify common coverage patterns.
- You want to summarize a large body of regulatory or legal text into digestible themes.
- You do NOT already have labeled categories -- if you do, use supervised text classification instead.

## How it works (intuitive)

### The core idea

Imagine you have a pile of 10,000 insurance claims notes. You dump all the words onto a table. You notice that some words hang out together frequently: "water," "pipe," "flood," "basement" tend to co-occur. Another cluster is "vehicle," "intersection," "rear-end," "bumper." A third might be "slip," "floor," "wet," "fracture." Each of these word clusters is a **topic**.

Topic modeling formalizes this intuition. Every document is treated as a mixture of topics, and every topic is a mixture of words.

### LDA: The Generative Story

Latent Dirichlet Allocation imagines that documents were written by the following (fictional) process:

1. **Pick a topic recipe for the document.** Before writing, the author decides "this note will be 70% about water damage and 30% about property repair." This recipe is drawn from a Dirichlet distribution (think of it as a way to generate random proportions that sum to 1).

2. **For each word in the document:**
   - First, flip a weighted coin to pick a topic according to the recipe.
   - Then, pick a word from that topic's vocabulary (each topic has its own probability distribution over all words).

3. The algorithm never sees the topics directly. It only sees the finished documents. **LDA works backward**: given the documents, it infers the most likely topics and document-topic mixtures that could have generated the observed words.

**Analogy**: Think of a buffet with K stations (topics). Each diner (document) fills their plate with food from multiple stations in different proportions. You only see the filled plates (the documents). LDA tries to figure out what was at each station and how much each diner took from each one.

### NMF: The Matrix Approach

Non-negative Matrix Factorization takes a more algebraic route:

1. Build a **term-document matrix** (or, more commonly, a TF-IDF matrix) where rows are documents and columns are words. Each cell holds the (weighted) count of that word in that document.

2. Factor this matrix into two smaller matrices:
   - **W** (documents x topics): How much each document relates to each topic.
   - **H** (topics x words): How much each word relates to each topic.

3. The constraint is that all values in W and H must be non-negative (no negative word frequencies), which naturally produces parts-based, interpretable decompositions.

4. The algorithm iteratively adjusts W and H to minimize the reconstruction error between the original matrix and W x H.

**Analogy**: Imagine a recipe book (the original matrix). NMF tries to decompose it into a small set of "base recipes" (topics) and a set of proportions telling you how much of each base recipe makes up each actual recipe.

### LDA vs. NMF at a glance

| Aspect | LDA | NMF |
|--------|-----|-----|
| Foundation | Probabilistic generative model | Linear algebra (matrix factorization) |
| Output | Probability distributions | Non-negative factor loadings |
| Speed | Slower (especially variational or Gibbs) | Faster on moderate-sized corpora |
| Interpretability | Generally good | Often sharper/crisper topics |
| Streaming | Supported (online LDA) | Harder to do incrementally |
| Requires | Bag-of-words (count) matrix | TF-IDF matrix (preferred) |

## Key concepts

### Text preprocessing pipeline

Before any topic model, you need clean input:

1. **Tokenization**: Split text into individual words.
2. **Lowercasing**: Normalize case.
3. **Stop word removal**: Remove common words ("the", "is", "and") that carry no topical meaning.
4. **Lemmatization or stemming**: Reduce words to their root form ("running" -> "run", "policies" -> "policy").
5. **Bigrams/trigrams** (optional): Detect multi-word phrases ("water damage", "bodily injury") using collocation detection.
6. **Build a dictionary and corpus**: Map words to integer IDs and convert documents to bag-of-words or TF-IDF representations.

### LDA mathematical formulation

For each of $K$ topics:

$$\phi_k \sim \text{Dirichlet}(\beta)$$

This gives each topic $k$ a distribution over the vocabulary. For each document $d$:

$$\theta_d \sim \text{Dirichlet}(\alpha)$$

This gives each document a distribution over topics. For each word position $n$ in document $d$:

$$z_{d,n} \sim \text{Multinomial}(\theta_d) \quad \text{(pick a topic)}$$
$$w_{d,n} \sim \text{Multinomial}(\phi_{z_{d,n}}) \quad \text{(pick a word from that topic)}$$

Where:
- $\alpha$ controls document-topic sparsity. Low $\alpha$ means each document is dominated by few topics. High $\alpha$ means documents are mixtures of many topics.
- $\beta$ controls topic-word sparsity. Low $\beta$ means each topic is defined by few key words. High $\beta$ means topics use broader vocabulary.

### NMF mathematical formulation

Given a document-term matrix $V$ of shape $(m \times n)$:

$$V \approx W \cdot H$$

Where:
- $W$ has shape $(m \times k)$ -- document-topic weights
- $H$ has shape $(k \times n)$ -- topic-word weights
- $k$ is the number of topics (a hyperparameter you choose)

The optimization minimizes a divergence measure, typically:

$$\min_{W,H \geq 0} \| V - WH \|_F^2$$

or the generalized Kullback-Leibler divergence.

### Topic coherence

Topic coherence measures how semantically consistent the top words of a topic are. The idea: if a topic's top words frequently co-occur in real text, the topic is coherent.

Common metrics:
- **C_v coherence**: Combines normalized pointwise mutual information (NPMI) with cosine similarity. Generally the best single metric for human-judged topic quality.
- **C_umass**: Based on document co-occurrence counts. Faster but less reliable.
- **Intrusion test** (manual): Show a human the top 5 words of a topic plus one "intruder" word. If humans can easily spot the intruder, the topic is coherent.

### Choosing the number of topics (K)

This is the hardest part. There is no single right answer. Strategies:

1. **Coherence score curve**: Train models for K = 5, 10, 15, 20, ..., 50. Plot coherence vs. K. Look for an "elbow" or peak.
2. **Perplexity** (LDA only): Measures how well the model predicts held-out documents. Lower is better, but perplexity does not always correlate with human interpretability.
3. **Domain knowledge**: If you know there are roughly 8 types of claims in your book of business, start around K = 8-12 and iterate.
4. **Hierarchical approach**: Start with a small K for broad themes, then sub-model specific topics for finer granularity.

## Python implementation

### LDA with Gensim

```python
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# Assume `processed_docs` is a list of lists of tokens
# e.g., [["water", "damage", "pipe", "basement"], ["vehicle", "collision", ...], ...]

# Build dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=5, no_above=0.5)  # Remove rare/common words
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train LDA
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    random_state=42,
    passes=15,          # Number of passes through the corpus
    alpha='auto',       # Learn alpha from data
    eta='auto',         # Learn beta from data
    per_word_topics=True
)

# Print topics
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")

# Evaluate coherence
coherence_model = CoherenceModel(
    model=lda_model,
    texts=processed_docs,
    dictionary=dictionary,
    coherence='c_v'
)
print(f"Coherence Score: {coherence_model.get_coherence():.4f}")
```

### NMF with scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Assume `raw_documents` is a list of strings
vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=5000,
    stop_words='english'
)
tfidf_matrix = vectorizer.fit_transform(raw_documents)

# Train NMF
nmf_model = NMF(
    n_components=10,
    random_state=42,
    max_iter=500,
    alpha_W=0.1,     # Regularization
    l1_ratio=0.5
)
W = nmf_model.fit_transform(tfidf_matrix)
H = nmf_model.components_

# Print topics
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(H):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

### Choosing K with coherence grid search

```python
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary,
                         num_topics=num_topics, random_state=42, passes=10)
        model_list.append(model)
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(cm.get_coherence())
    return model_list, coherence_values
```

## R implementation (if applicable)

R has the `topicmodels` and `text2vec` packages for LDA/NMF. The `tidytext` package integrates well for preprocessing:

```r
library(topicmodels)
library(tidytext)

# Create a DocumentTermMatrix from your text data
dtm <- DocumentTermMatrix(corpus)

# Fit LDA
lda_model <- LDA(dtm, k = 10, control = list(seed = 42))

# Extract topics
topics <- tidy(lda_model, matrix = "beta")
```

Python (Gensim + scikit-learn) is generally preferred for topic modeling due to richer tooling, faster iteration, and better visualization support (pyLDAvis).

## Gotchas & lessons learned

### Preprocessing is 80% of the work
Bad preprocessing kills topic quality. If you skip lemmatization, "policy" and "policies" become separate words and dilute your topics. If you leave in stop words, every topic will be dominated by "the," "and," "is."

### Domain-specific stop words matter
Standard English stop word lists do not include industry jargon that appears everywhere. In insurance claims notes, words like "claim," "insured," "claimant," and "loss" may appear in nearly every document. These should be added to your custom stop word list -- otherwise every topic will include them and they will crowd out the meaningful words.

### More topics is not always better
Adding more topics eventually leads to fragmented, redundant, or incoherent topics. If topic 7 and topic 12 look nearly identical, you probably have too many. If one topic seems to contain three unrelated themes, you probably need more.

### LDA is sensitive to hyperparameters
The alpha and beta (eta) parameters matter. Use `alpha='auto'` in Gensim to let it learn from data. If you set alpha too high, every document looks like a uniform mixture of all topics, which defeats the purpose.

### NMF tends to give crisper topics
In practice, NMF with TF-IDF often produces more distinct, human-readable topics than LDA, especially on shorter documents. Start with NMF for quick exploration, then try LDA for probabilistic analysis.

### Short documents are challenging
Topic models assume documents contain enough words to exhibit a topic mixture. Very short texts (tweets, one-line claim descriptions) often lack enough signal. Consider aggregating short texts (e.g., all notes for one claim) before modeling.

### Reproducibility
LDA with Gibbs sampling is stochastic. Set random seeds and run multiple initializations. Compare results across runs to identify stable topics.

### Do not over-interpret word rankings
The top 10 words of a topic are a summary, not a definition. Always look at representative documents for each topic to understand what the topic really captures.

### Visualization
Use **pyLDAvis** for interactive LDA topic exploration. It shows inter-topic distances and word relevance, and it is invaluable for tuning K and understanding topic overlap.

```python
import pyLDAvis.gensim_models
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)
```

## Resources

- [Gensim documentation](https://radimrehurek.com/gensim/) -- The primary Python library for LDA and related models
- [scikit-learn NMF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [Blei, Ng, Jordan (2003) - Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) -- The original LDA paper
- [pyLDAvis](https://github.com/bmabey/pyLDAvis) -- Interactive visualization for LDA models
- [Topic Model Evaluation in Python (Towards Data Science)](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0)
- [Lee and Seung (1999) - Learning the Parts of Objects by Non-negative Matrix Factorization](https://www.nature.com/articles/44565) -- The foundational NMF paper
- [Machine Learning Mastery - Topic Modeling](https://machinelearningmastery.com/topic-modeling-with-gensim-python/)
- [Actuaries Digital - Text Analytics in Insurance](https://www.actuaries.digital/) -- Industry applications
