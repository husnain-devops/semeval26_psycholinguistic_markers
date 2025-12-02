# EDA Pipeline Documentation

This document provides a comprehensive guide to all scripts in the EDA pipeline, the methods they use, the results they produce, and how to interpret those results.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Script Documentation](#script-documentation)
   - [data_ingest.py](#1-data_ingestpy)
   - [preprocessing.py](#2-preprocessingpy)
   - [lexical_complexity.py](#3-lexical_complexitypy)
   - [readability_entropy.py](#4-readability_entropypy)
   - [discourse_markers_and_hedges.py](#5-discourse_markers_and_hedgespy)
   - [sentiment_emotion.py](#6-sentiment_emotionpy)
   - [marker_analysis.py](#7-marker_analysispy)
   - [topic_modeling.py](#8-topic_modelingpy)
   - [embeddings_and_clustering.py](#9-embeddings_and_clusteringpy)
   - [statistical_tests_and_feature_importance.py](#10-statistical_tests_and_feature_importancepy)
   - [reporting.py](#11-reportingpy)

---

## Pipeline Overview

The EDA pipeline processes Reddit posts to analyze conspiracy theory content. It follows this sequence:

1. **Data Ingestion** → Clean and normalize input data
2. **Preprocessing** → Extract basic linguistic features
3. **Feature Extraction** → Compute various text metrics
4. **Analysis** → Statistical tests and modeling
5. **Reporting** → Generate summary reports

All scripts log their progress to `reports/run_log.txt` and save outputs to organized directories.

---

## Script Documentation

### 1. data_ingest.py

**Purpose**: Load and normalize raw JSONL data into a clean CSV format.

**EDA Methods**:
- Data normalization (label standardization)
- Basic descriptive statistics
- Data validation and cleaning

**Input**:
- `data_raw/train_rehydrated.jsonl` (or `train_rehydrated.jsonl` in repo root)

**Output Files**:
- `data_processed/data_clean.csv`: Cleaned dataset with normalized labels
- `data_processed/dataset_summary.json`: Dataset statistics

**Output Columns** (data_clean.csv):
- `_id`: Document identifier
- `text`: Post text content
- `subreddit`: Subreddit name
- `conspiracy`: Normalized label (`yes`, `no`, `cant_tell`)
- `markers`: JSON string of discourse markers

**How to Interpret**:
- **dataset_summary.json**: Contains:
  - `total_rows`: Total number of documents
  - `distribution`: Count of each label (yes/no/cant_tell)
  - `marker_counts`: Count of each marker type found
  - `mean_words`, `median_words`, `min_words`, `max_words`: Text length statistics
  - `seed`: Random seed used (42)

**Example Interpretation**:
{
  "total_rows": 4316,
  "distribution": {"no": 1990, "yes": 1541, "cant_tell": 785},
  "marker_counts": {"Actor": 1200, "Action": 800, ...}
}This shows the dataset has 4316 documents with an imbalanced distribution favoring "no" labels.

---

### 2. preprocessing.py

**Purpose**: Extract basic linguistic and structural features from text.

**EDA Methods**:
- Text normalization (Unicode NFKC, whitespace cleaning)
- Part-of-Speech (POS) tagging using spaCy
- Token counting and sentence segmentation
- Punctuation analysis
- URL/email detection

**Input**:
- `data_processed/data_clean.csv`

**Output Files**:
- `data_processed/data_processed.parquet` (or `.csv` if parquet fails)
- `features/token_pos_counts.csv`: Token and POS statistics per document

**Output Columns** (token_pos_counts.csv):
- `_id`: Document identifier
- `n_tokens`: Total word count
- `n_sentences`: Number of sentences
- `n_nouns`: Count of nouns (including proper nouns)
- `n_verbs`: Count of verbs
- `n_adjs`: Count of adjectives
- `n_advs`: Count of adverbs
- `n_pronouns`: Count of pronouns

**Additional Features** (in data_processed.parquet/csv):
- `url_count`: Number of URLs in text
- `email_count`: Number of email addresses
- `stopword_ratio`: Proportion of stopwords (common words like "the", "a")
- `commas`, `qmarks`, `exclaims`, `ellipses`: Punctuation counts
- `allcaps_ratio`: Proportion of words in ALL CAPS

**How to Interpret**:
- **High `stopword_ratio`**: More common words, potentially simpler language
- **High `n_nouns`**: More entities/concepts mentioned
- **High `allcaps_ratio`**: More emphasis/emotional content
- **High `url_count`**: More external references
- Compare these metrics across `conspiracy` labels to find patterns

**Example**: A post with high `n_nouns` and `url_count` might be more factual/reference-heavy, while high `exclaims` and `allcaps_ratio` suggests emotional content.

---

### 3. lexical_complexity.py

**Purpose**: Compute lexical richness and vocabulary diversity metrics.

**EDA Methods**:
- Type-Token Ratio (TTR)
- Hapax legomena ratio
- Average word length
- Shannon entropy (vocabulary diversity)

**Input**:
- `data_processed/data_processed.parquet` (or `.csv`)

**Output Files**:
- `features/lexical_complexity.csv`: Per-document lexical metrics
- `features/lexical_summary_by_label.csv`: Summary statistics by label
- `features/lexical_summary_by_label.json`: JSON summary with mean/std
- `figures/lexical_summary_by_label.png`: Box plots comparing metrics by label

**Output Columns** (lexical_complexity.csv):
- `_id`: Document identifier
- `ttr`: Type-Token Ratio (unique words / total words)
  - Range: 0-1, higher = more diverse vocabulary
- `hapax_ratio`: Proportion of words that appear only once
  - Higher = more unique/rare words
- `avg_word_len`: Average character length per word
  - Higher = longer, potentially more complex words
- `shannon_entropy`: Information entropy of word distribution
  - Higher = more uniform word usage (less repetition)

**How to Interpret**:
- **TTR**: 
  - High TTR (>0.7): Diverse vocabulary, less repetition
  - Low TTR (<0.5): Repetitive language, limited vocabulary
- **Hapax Ratio**:
  - High: Many unique words, potentially technical/specialized
  - Low: Common vocabulary, more accessible
- **Shannon Entropy**:
  - High: Words used more evenly (less repetition)
  - Low: Some words dominate (more repetitive)

**Statistical Summary**:
The `lexical_summary_by_label.csv` provides descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max) for each metric grouped by conspiracy label.

**Plot Interpretation** (`lexical_summary_by_label.png`):
- Box plots show distribution of each metric across labels
- Compare medians (center line) to see which labels use more diverse vocabulary
- Wider boxes = more variation within that label

**Example**: If "yes" label has higher TTR and hapax_ratio than "no", conspiracy posts may use more diverse/unique vocabulary.

---

### 4. readability_entropy.py

**Purpose**: Compute readability scores using standard formulas.

**EDA Methods**:
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- Automated Readability Index (ARI)
- Coleman-Liau Index

**Input**:
- `data_processed/data_processed.parquet` (or `.csv`)

**Output Files**:
- `readability/readability_scores.csv`: Per-document readability scores

**Output Columns**:
- `_id`: Document identifier
- `flesch_reading_ease`: 0-100 scale (higher = easier)
  - 90-100: Very easy (5th grade)
  - 60-70: Standard (8th-9th grade)
  - 0-30: Very difficult (college graduate)
- `flesch_kincaid_grade`: U.S. grade level (0-18+)
- `gunning_fog`: Grade level (higher = more complex)
- `smog`: Grade level estimate
- `ari`: Grade level (0-14+)
- `coleman_liau`: Grade level

**How to Interpret**:
- **Lower scores** = More difficult/complex text
- **Higher scores** = Easier/more accessible text
- Compare across labels: Do conspiracy posts use simpler or more complex language?
- **Flesch Reading Ease** is most intuitive: 60-70 is average readability

**Example**: If "yes" posts have lower Flesch scores, they may use more complex language or longer sentences.

---

### 5. discourse_markers_and_hedges.py

**Purpose**: Count hedging language, certainty markers, discourse markers, and passive voice.

**EDA Methods**:
- Pattern matching for specific word/phrase lists
- Regular expression matching for passive voice

**Input**:
- `data_processed/data_processed.parquet` (or `.csv`)

**Output Files**:
- `features/discourse_markers.csv`: Per-document marker counts

**Output Columns**:
- `_id`: Document identifier
- `hedge_count`: Count of hedging words (maybe, perhaps, might, seems, possibly, may)
  - Higher = more uncertain/qualifying language
- `certainty_count`: Count of certainty words (definitely, certainly, proof, proven, prove, undeniable, obviously)
  - Higher = more assertive/confident language
- `discourse_count`: Count of discourse markers (actually, in fact, the truth, frankly, to be honest)
  - Higher = more explicit framing/positioning
- `passive_count`: Count of passive voice constructions (was/were/is/are + past participle)
  - Higher = more passive voice (may indicate distancing or objectivity)

**How to Interpret**:
- **Hedge vs. Certainty**: 
  - High hedge, low certainty = tentative/uncertain tone
  - Low hedge, high certainty = confident/assertive tone
- **Discourse markers**: Indicate explicit framing or emphasis
- **Passive voice**: May indicate objectivity or distancing from claims

**Example**: Conspiracy posts might have higher certainty_count and lower hedge_count, suggesting more confident assertions.

---

### 6. sentiment_emotion.py

**Purpose**: Compute sentiment scores using VADER (Valence Aware Dictionary and sEntiment Reasoner).

**EDA Methods**:
- VADER sentiment analysis (rule-based, optimized for social media)

**Input**:
- `data_processed/data_processed.parquet` (or `.csv`)

**Output Files**:
- `features/sentiment_emotion.csv`: Per-document sentiment scores
- `features/sentiment_by_label.csv`: Mean sentiment by label

**Output Columns** (sentiment_emotion.csv):
- `_id`: Document identifier
- `neg`: Negative sentiment score (0-1, higher = more negative)
- `neu`: Neutral sentiment score (0-1, higher = more neutral)
- `pos`: Positive sentiment score (0-1, higher = more positive)
- `compound`: Overall sentiment score (-1 to +1)
  - > 0.05: Positive
  - < -0.05: Negative
  - Between: Neutral

**How to Interpret**:
- **Compound score** is most useful for overall sentiment
- **High negative + low positive**: Negative emotional tone
- **High positive**: Positive/optimistic tone
- **High neutral**: Factual/objective tone

**sentiment_by_label.csv**: Shows average compound sentiment for each label, useful for comparing emotional tone across conspiracy vs. non-conspiracy posts.

**Example**: If "yes" posts have more negative compound scores, conspiracy content may be more emotionally negative.

---

### 7. marker_analysis.py

**Purpose**: Analyze discourse marker annotations (Actor, Action, Victim, Evidence, Effect).

**EDA Methods**:
- Frequency counting
- Co-occurrence analysis
- Heatmap visualization

**Input**:
- `data_processed/data_clean.csv`

**Output Files**:
- `features/marker_counts.csv`: Count of each marker type
- `features/marker_cooccurrence.csv`: Co-occurrence matrix
- `figures/marker_counts.png`: Bar chart of marker frequencies
- `figures/marker_cooccurrence_heatmap.png`: Heatmap of co-occurrences

**Output Columns** (marker_counts.csv):
- `marker`: Marker type (Actor, Action, Victim, Evidence, Effect)
- `count`: Number of documents containing this marker type

**marker_cooccurrence.csv**: Symmetric matrix showing how often marker types appear together in the same document.

**How to Interpret**:
- **marker_counts.csv**: Which marker types are most common?
  - High Actor count: Many posts identify actors
  - High Evidence count: Many posts cite evidence
- **Co-occurrence matrix**: Which markers appear together?
  - High values on diagonal: Markers often co-occur
  - Example: High Actor-Evidence co-occurrence suggests posts that identify actors also cite evidence

**Plot Interpretation**:
- **marker_counts.png**: Bar chart showing frequency of each marker type
- **marker_cooccurrence_heatmap.png**: Color intensity shows co-occurrence strength
  - Darker = more frequent co-occurrence
  - Diagonal shows self-co-occurrence (always 0)

**Example**: If Actor and Evidence co-occur frequently, conspiracy posts may systematically link actors to evidence.

---

### 8. topic_modeling.py

**Purpose**: Discover latent topics in the corpus using LDA and BERTopic.

**EDA Methods**:
- Latent Dirichlet Allocation (LDA) - probabilistic topic modeling
- BERTopic - transformer-based topic modeling
- Topic visualization (pyLDAvis)

**Input**:
- `data_processed/data_clean.csv`

**Output Files**:
- `topics/lda_topics_8.csv`, `lda_topics_12.csv`, `lda_topics_20.csv`: LDA topics for different numbers of topics (k)
- `topics/lda_vis_8.html`, `lda_vis_12.html`, `lda_vis_20.html`: Interactive topic visualizations
- `topics/bertopic_topics.csv`: BERTopic topic information
- `topics/bertopic_doc_topics.csv`: Document-to-topic assignments

**Output Columns** (lda_topics_*.csv):
- `topic_id`: Topic identifier (0 to k-1)
- `words`: Top 15 words for this topic (space-separated)

**bertopic_topics.csv**: Contains topic information including:
- Topic ID, count, name, and representative documents

**bertopic_doc_topics.csv**:
- `_id`: Document identifier
- `topic`: Assigned topic ID (-1 = outlier/no topic)

**How to Interpret**:
- **LDA Topics**: 
  - Each topic is a distribution over words
  - Top words indicate what the topic is about
  - Compare topics across different k values to find optimal number
- **Topic Assignment**: 
  - Documents assigned to topics show thematic clustering
  - Analyze which topics are associated with "yes" vs "no" labels
- **pyLDAvis HTML**: Interactive visualization showing:
  - Topic-word relationships
  - Inter-topic distances
  - Most relevant terms per topic

**Example**: If Topic 5 has words like "government", "cover", "secret" and is associated with "yes" labels, it may represent a government conspiracy theme.

---

### 9. embeddings_and_clustering.py

**Purpose**: Generate semantic embeddings and perform dimensionality reduction and clustering.

**EDA Methods**:
- Sentence transformers (all-mpnet-base-v2)
- UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
- HDBSCAN (Hierarchical Density-Based Spatial Clustering) for clustering

**Input**:
- `data_processed/data_clean.csv`

**Output Files**:
- `embeddings/embeddings.npy`: Dense vector embeddings (768 dimensions per document)
- `embeddings/doc_clusters.csv`: Cluster assignments per document
- `figures/umap_clusters.png`: 2D visualization of clusters

**Output Columns** (doc_clusters.csv):
- `_id`: Document identifier
- `cluster`: Cluster ID (-1 = noise/outlier, 0+ = cluster ID)

**How to Interpret**:
- **Embeddings**: Dense vector representations capturing semantic meaning
  - Similar documents have similar embeddings
  - Can be used for similarity search or classification
- **UMAP Visualization**: 2D projection showing:
  - Clusters as groups of points
  - Outliers as isolated points
  - Semantic relationships (closer = more similar)
- **Cluster Analysis**:
  - Cluster -1: Noise/outliers (don't fit any cluster)
  - Cluster 0+: Thematic groups
  - Analyze which clusters contain "yes" vs "no" labels

**Plot Interpretation** (`umap_clusters.png`):
- Points colored by cluster ID
- Tight groups = coherent themes
- Scattered points = diverse content
- Compare cluster composition with conspiracy labels

**Example**: If Cluster 3 contains mostly "yes" labels, it may represent a specific conspiracy theme that can be further analyzed.

---

### 10. statistical_tests_and_feature_importance.py

**Purpose**: Perform statistical tests and train predictive models to identify important features.

**EDA Methods**:
- Independent samples t-test (comparing "yes" vs "no" groups)
- Logistic Regression with cross-validation
- Random Forest classifier
- SHAP (SHapley Additive exPlanations) for feature importance

**Input**:
- `data_processed/data_processed.parquet` (or `.csv`)
- Merges: `lexical_complexity.csv`, `discourse_markers.csv`, `sentiment_emotion.csv`, `token_pos_counts.csv`

**Output Files**:
- `features/univariate_tests.csv`: T-test results for each feature
- `figures/univariate_tests.png`: Visualization of t-statistics and p-values
- `models/rf_model.joblib`: Trained Random Forest model
- `figures/shap_summary.png`: SHAP summary plot
- `shap_top20.csv`: Top 20 features by SHAP importance

**Output Columns** (univariate_tests.csv):
- `feature`: Feature name
- `tstat`: T-statistic (positive = "yes" group higher, negative = "no" group higher)
- `p`: P-value (lower = more significant difference)
  - < 0.05: Statistically significant
  - < 0.01: Highly significant
  - < 0.001: Very highly significant

**shap_top20.csv**:
- `feature`: Feature name
- `importance`: Mean absolute SHAP value (higher = more important for prediction)

**How to Interpret**:
- **T-tests**:
  - Significant p-value (< 0.05) = feature differs between groups
  - Large |tstat| = bigger difference between groups
  - Positive tstat = "yes" group has higher values
  - Negative tstat = "no" group has higher values
- **SHAP Values**:
  - Higher importance = feature more predictive
  - Positive SHAP = increases probability of "yes"
  - Negative SHAP = decreases probability of "yes"
- **Model Performance**:
  - LogReg F1 score: Cross-validated F1 score (0-1, higher = better)
  - F1 > 0.7: Good performance
  - F1 > 0.8: Very good performance

**Plot Interpretation**:
- **univariate_tests.png**: 
  - Left: T-statistics (features with large |tstat| are most different)
  - Right: P-values on log scale (features below 0.05 line are significant)
- **shap_summary.png**: 
  - Features ranked by importance
  - Color shows feature value (red = high, blue = low)
  - Position shows impact on prediction

**Example**: If `certainty_count` has high SHAP importance and positive t-statistic, conspiracy posts ("yes") use more certainty words, and this is a strong predictor.

---

### 11. reporting.py

**Purpose**: Generate a summary markdown report aggregating key findings.

**EDA Methods**:
- Data aggregation
- Report generation

**Input**:
- `data_processed/dataset_summary.json`
- `shap_top20.csv` (if available)
- `topics/bertopic_topics.csv` (if available)
- `topics/lda_topics_12.csv` (if available)

**Output Files**:
- `reports/EDA_report.md`: Markdown summary report

**Report Contents**:
1. **Dataset Summary**: Basic statistics from dataset_summary.json
2. **Top Features by SHAP**: Most important predictive features
3. **BERTopic Topics**: Discovered topics from BERTopic
4. **LDA Topics**: Discovered topics from LDA (k=12)

**How to Interpret**:
- **EDA_report.md**: Quick reference summary of key findings
- Use as a starting point for deeper analysis
- Combines results from multiple analysis steps

---

## Output Directory Structure

```
EDA-Rehydrated/
├── data_processed/          # Cleaned and processed data
├── features/                # Extracted features (CSV files)
├── figures/                 # Visualizations (PNG files)
├── topics/                 # Topic modeling results
├── embeddings/             # Document embeddings
├── readability/            # Readability scores
├── models/                 # Trained models
└── reports/                # Logs and summary reports
```

---

## Interpreting Results: Quick Guide

### Comparing Labels

Most analyses compare "yes" (conspiracy) vs "no" (non-conspiracy) vs "cant_tell":

1. **Statistical Tests**: Look for features with p < 0.05 in `univariate_tests.csv`
2. **Feature Importance**: Check `shap_top20.csv` for most predictive features
3. **Visualizations**: Compare distributions in box plots and histograms
4. **Topic Analysis**: See which topics are associated with "yes" labels

### Key Questions to Answer

1. **Language Complexity**: Do conspiracy posts use simpler or more complex language?
   - Check: `lexical_complexity.csv`, `readability_scores.csv`

2. **Emotional Tone**: Are conspiracy posts more negative?
   - Check: `sentiment_emotion.csv`, `sentiment_by_label.csv`

3. **Certainty**: Do conspiracy posts use more certainty words?
   - Check: `discourse_markers.csv` (certainty_count vs hedge_count)

4. **Structure**: Do conspiracy posts have different linguistic structure?
   - Check: `token_pos_counts.csv` (POS tag distributions)

5. **Themes**: What topics/themes are associated with conspiracy content?
   - Check: `topics/` files, `embeddings/doc_clusters.csv`

---

## Troubleshooting

### Missing Outputs
- Check `reports/run_log.txt` for error messages
- Some scripts require optional dependencies (e.g., VADER, textstat)
- Scripts will log missing dependencies and continue

### Empty Files
- Verify input files exist and are readable
- Check encoding issues (some scripts handle UTF-8, latin-1, or error replacement)

### Plot Generation Failures
- Ensure matplotlib and seaborn are installed
- Check that `figures/` directory exists and is writable

---

## Next Steps

After running the pipeline:

1. **Review Visualizations**: Start with `figures/` directory
2. **Check Statistical Tests**: Look at `univariate_tests.csv` for significant differences
3. **Examine Feature Importance**: Review `shap_top20.csv` for predictive features
4. **Explore Topics**: Use pyLDAvis HTML files for interactive topic exploration
5. **Read Summary Report**: Check `reports/EDA_report.md` for aggregated findings

For interactive exploration, use the Jupyter notebooks:
- `notebooks/01_EDA_quicklook.ipynb`: Quick visualizations
- `notebooks/02_Feature_Analysis.ipynb`: Feature analysis and distributions
