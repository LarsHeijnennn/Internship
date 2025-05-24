## The goal and Big Picture

Our main goal here is to understand why a relatively simple Linear Regression model can predict sentence length so well using seemingly complex BERT embeddings. 
The hypothesis is that the BERT embeddings, despite their high dimensionality (768 dimensions), must be organizing sentences in such a way that sentence length is a somewhat "separable" feature.

Hierarchical clustering is a way to explore the structure of the data, which are the sentence embeddings, by grouping similar embeddings together. A dendogram is a visual representation of this grouping process. If we see that sentences with similar lengths naturally fall into the same or nearby clusters in the dendrogram, it would visually support the idea that the embeddings capture sentence length information in a structured way.

## The input: Sentence Embeddings
The input is the ```X_sentence_features```, which is a NumPy array where each row represet a sentence from the ```filtered_cleaned_sentences``` list. Each row contains 768 numbers, which are the BERT embedding for that sentence. It represent the meaning and context of the sentence, as learned by BERT. The idea is that sentences with similar meanings or structures will have embeddings that are "close" to each other in this 768 dimensional space. The input has a shape of (3868, 768), meaning that we have 3868 sentences, with a dimension of 768.

## The process: hierarchical clustering with Ward's method
```linked = linkage(X_sentence_features, method='ward')```

**Hierarchical Clustering Basics**: This is an bottom-up approach. It starts by treating each of the 3868 sentence embeddings as its own individual cluster. Then, it iteratively merges the two "closest" or most similar clusters into a new, larger cluster. This process continues until all sentences belong to a single, giant cluster. 

Ward's method is a method that decides which two clusters to merge at each step, aiming to minimize the total within-cluster variance. The method chooses to merge that leads to the smallest increase in this total variance. 

The resulting ```linked``` matrix does not contain cluster assignments directly. Instead, it's a record of all the merges that happened. Each row in this matrix describes one merge operation:
    1. Index of the first cluster being merged.
    2. Index of the second cluster being merged.
    3. The distance between these two clusters when they were merged (using Ward's criterion, this distance is related to the increase in variance).
    4. The number of original data points in the newly formed cluster.

The output: The Dendrogram Plot
```dendrogram(linked, ...)``` --> The dendrogram function takes the linked matrix (the history of merges) and visualizes it as a tree diagram.

### How to read?
**Y-axis (distance (Ward))**:  This axis shows how different clusters were when they were joined. When two branches connect, the height of the line shows how far apart they were. The higher the line, the more different the clusters were before merging.
**X-axis (Sample Index (or Cluster Size if contracted))**: 
    - At the bottom, if the dendrogram weren't truncated, each "leaf" would represent one of your original 3868 sentences.
    - Because plotting 3868 leaves would be dense and unreadable, I've gone ahead and truncated the dendrogram to show only the formation of the "last p" (in this case, 30) merged clusters. This means that we look at the top part of the tree, showing how larger clusters merge.
    - ```show_contracted=True```: This visually shortens branches that represent many individual merges happening at lower distances, making the upper structure clearer. The little dots you see at the bottom of some branches represent these contracted (hidden) merges.
- Each horizontal lines signifies a merge. Two clusters (as seen by the vertical lines below it) are joined at the distance indicated by the height of this horizontal line. 
- Vertical lines represent the clusters themselves. The length might give a visual cue abot how distinct a cluster was before it got merged into a larger one. 
- Different colors for branches are purely for visual distinction. 
- The red dashed line is an illustrative line I added. If you were to "cut: the dendrogram horizontally, every branch that intersects the cut would become a separate, "flat" cluster. If you lower the line, you would "cut" more, smaller, and potentially more specific clusters. It is a line to decide how many clusters you want the data to be divided into. 

## What does this dendrogram potentially tell us?
It shows how sentence embeddings group together bases on siilarity. There are some large grouping, like at a high level at around a distance of 150.  As you go down, these split into smaller, presumably more coherent sub-clusters. 

If sentences of similar lenghts also have similar BERT embeddings, we might expect that the clusters formed would, to some extent, group sentences by length. For example, one large branch might predominantly contain shorter sentences, and another might contain longer sentences. 

If the embeddings naturally cluster in a way that aligns with sentence length, it makes it easier for a linear model to find such a separating hyperplane.