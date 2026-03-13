## Online Forum Bot Detection
Concerned about misinformation on online forums, I'm wondering if I could use machine learning to detect bots and targeted misinformation campaings myself.

## Forum overview
On <code>$online_form</code> (name redacted), people can upload media, and after creating an account, users can comment on posts and leave a positive or negative "like" on both posts and comments. Users can also reply to comments, creating a tree structure of comments.
The form will not be disclosed for privacy reasons. If we can Identify any misinformation campaings, they will be reported to the form moderators and any relevant authorities. The aim is not to cause harm or negative PR to the forum or its users, but to promote a healthier online environment.

### Data collection
We collected 19M comments for ±300K posts, rangeing from 2006 to end 2025. There were ±34K unique authors. 
The data was collected using the forum's public API. 

All data was stored in a local duckdb database for efficient analysis.

## Feature engineering
Determining the features to discriminate bots from human traffic is hard. My assumptions are:
- Bots will post more repetitively (e.g. copy-pasting the same comment multiple times) and thus have less lexical diversity (use less different words) and repeat the same words more often
- Bots will post more links (e.g. to external misinformation sources) and thus have a higher link ratio.
- Bots will post more negative comments (e.g. to create conflict and engagement) and thus have a lower sentiment polarity.
- Bots will post more often and in bursts (e.g. posting 100 comments in 1 hour, then nothing for a week).
- Bots will post more often at specific hours (e.g. when the misinformation campaign is most effective) and thus have a lower hour entropy.

Those assumptions led to the following features being calculated for each user:

- Lexical Diversity
*Ratio of unique words to total words across all comments written by a user.*

- Repeated Comment Ratio
*Proportion of duplicated comment texts posted by a user, used as an indicator of repeated posting.*

- Link Ratio
*Fraction of comments containing URLs (identified via the substring "http").*

- Sentiment Polarity
*Average sentiment polarity across a user's comments, computed using the TextBlob library.*

- Comments per Day
*Average number of comments posted per day since the user's first recorded comment.*

- Burstiness (hours)
*Standard deviation of time intervals between consecutive comments (in hours), capturing burst-like posting behavior.*

- Hour Entropy
*Entropy of the distribution of posting times across 24 hourly bins, measuring how evenly a user spreads activity throughout the day.*

## Implementation details
All features are calculated per author_id.
computations including sentiment analysis, temporal statistics, and grouping operations are parallelized using <code>ProcessPoolExecutor</code> to improve processing speed. Data is processed in chunks to efficiently handle large volumes of comments.

## Preprocessing
Before modeling:
- Missing values are replaced with zeros.
- All features are standardized using StandardScaler to ensure comparable scales across variables.

## Clustering (User segmentation)
User segmentation is performed using KMeans clustering on the standardized feature set.

## Model Selection
To determine the optimal number of clusters, the Silhouette Score is computed for values of 𝑘 ranging from 2 to <code>MAX_CLUSTERS</code>.
The value of 𝑘 that maximizes the silhouette score is selected.

## Output
Each user receives a segment label (0..k-1).
Cluster-level statistics (mean feature values and user counts).