def load_and_explore_data(file_path):
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    print("\nFirst 3 lyrics samples:")
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(df.iloc[i]['Lyric'][:200] + "..." if len(df.iloc[i]['Lyric']) > 200 else df.iloc[i]['Lyric'])
    
    return df


class TextPreprocessor:
    """A comprehensive text preprocessing class"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        text = text.lower()
        
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word not in self.stop_words])
    
    def lemmatize_text(self, text):
        """Lemmatize text"""
        tokens = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in tokens])
    
    def stem_text(self, text):
        """Stem text"""
        tokens = word_tokenize(text)
        return ' '.join([self.stemmer.stem(word) for word in tokens])
    
    def preprocess_text(self, text, use_lemmatization=True, remove_stopwords=True):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        if use_lemmatization:
            text = self.lemmatize_text(text)
        else:
            text = self.stem_text(text)
        
        return text

def preprocess_dataset(df, column_name='Lyric'):
    """Preprocess the entire dataset"""
    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)
    
    preprocessor = TextPreprocessor()
    
    # different preprocessing combinations
    preprocessing_methods = {
        'cleaned_only': lambda x: preprocessor.clean_text(x),
        'cleaned_no_stopwords': lambda x: preprocessor.remove_stopwords(preprocessor.clean_text(x)),
        'cleaned_lemmatized': lambda x: preprocessor.preprocess_text(x, use_lemmatization=True),
        'cleaned_stemmed': lambda x: preprocessor.preprocess_text(x, use_lemmatization=False),
    }
    
    for method_name, method_func in preprocessing_methods.items():
        print(f"\nApplying {method_name} preprocessing...")
        df[f'{column_name}_{method_name}'] = df[column_name].apply(method_func)
    
    print("\nPreprocessing Examples:")
    sample_text = df[column_name].iloc[0][:200] + "..." if len(df[column_name].iloc[0]) > 200 else df[column_name].iloc[0]
    print(f"Original: {sample_text}")
    
    for method_name in preprocessing_methods.keys():
        processed_text = df[f'{column_name}_{method_name}'].iloc[0][:200] + "..." if len(df[f'{column_name}_{method_name}'].iloc[0]) > 200 else df[f'{column_name}_{method_name}'].iloc[0]
        print(f"{method_name}: {processed_text}")
    
    return df

# Part 4: Feature Extraction using SentenceTransformers
def extract_features(texts, model_name='all-MiniLM-L6-v2', scale=True):
    """Extract features using SentenceTransformers and optionally scale"""
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Extracting features from text data...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    if scale:
        print("Scaling features...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    print(f"Feature extraction complete!")
    print(f"Feature vector shape: {embeddings.shape}")
    print(f"Each text is represented by {embeddings.shape[1]} features")
    
    return embeddings


# Part 5: Clustering Algorithms
class ClusteringAnalyzer:
    """Class to perform various clustering algorithms"""
    
    def __init__(self, features):
        self.features = features
        self.results = {}
        
    def elbow_method(self, max_k=10):
        """Find optimal K using elbow method"""
        print("\n" + "=" * 60)
        print("ELBOW METHOD FOR K-MEANS")
        print("=" * 60)
        
        inertias = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.features)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal K')
        plt.grid(True)
        plt.show()
        
        return k_range, inertias
    
    def kmeans_clustering(self, n_clusters=5):
        """Perform K-Means clustering"""
        print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.features)
        
        self.results['kmeans'] = {
            'labels': labels,
            'model': kmeans,
            'n_clusters': n_clusters
        }
        
        return labels
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering"""
        print(f"\nPerforming DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN found {n_clusters} clusters")
        
        self.results['dbscan'] = {
            'labels': labels,
            'model': dbscan,
            'n_clusters': n_clusters
        }
        
        return labels
    
    def hierarchical_clustering(self, n_clusters=5):
        """Perform Hierarchical clustering"""
        print(f"\nPerforming Hierarchical clustering with {n_clusters} clusters...")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(self.features)
        
        self.results['hierarchical'] = {
            'labels': labels,
            'model': hierarchical,
            'n_clusters': n_clusters
        }
        
        return labels

# Part 6: Dimensionality Reduction
def perform_pca(features, n_components=2):
    """Perform PCA for dimensionality reduction"""
    print("\n" + "=" * 60)
    print("DIMENSIONALITY REDUCTION (PCA)")
    print("=" * 60)
    
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features)
    
    print(f"Original features shape: {features.shape}")
    print(f"Reduced features shape: {features_reduced.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    return features_reduced, pca

# Part 7: Visualization Functions
def visualize_clusters(features_2d, labels, title, method_name):
    """Visualize clusters in 2D space"""
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:  # Noise points in DBSCAN
            plt.scatter(features_2d[labels == label, 0], 
                       features_2d[labels == label, 1], 
                       c='black', marker='x', s=50, label='Noise')
        else:
            plt.scatter(features_2d[labels == label, 0], 
                       features_2d[labels == label, 1], 
                       c=[color], s=50, label=f'Cluster {label}')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'{title} - {method_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Part 8: Evaluation Metrics
def calculate_metrics(features, labels):
    """Calculate clustering evaluation metrics"""
    
    # Remove noise points for silhouette score (DBSCAN)
    if -1 in labels:
        mask = labels != -1
        features_clean = features[mask]
        labels_clean = labels[mask]
    else:
        features_clean = features
        labels_clean = labels
    
    metrics = {}
    
    if len(np.unique(labels_clean)) > 1:
        silhouette = silhouette_score(features_clean, labels_clean)
        metrics['silhouette'] = silhouette
    else:
        metrics['silhouette'] = -1

    try:
        homogeneity = homogeneity_score(labels) 
        metrics['homogeneity'] = homogeneity
    except:
        metrics['homogeneity'] = -1 
        
    metrics['n_clusters'] = len(np.unique(labels)) - (1 if -1 in labels else 0)
    
    metrics['noise_points'] = np.sum(labels == -1)
    
    return metrics

def analyze_cluster_samples(df, labels, text_column, n_samples=2):
    """Analyze samples from each cluster"""
    print("\n" + "=" * 60)
    print("CLUSTER SAMPLE ANALYSIS")
    print("=" * 60)
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1: 
            continue
            
        cluster_indices = np.where(labels == label)[0]
        cluster_size = len(cluster_indices)
        
        print(f"\n--- Cluster {label} (Size: {cluster_size}) ---")
        
        sample_indices = np.random.choice(cluster_indices, 
                                        min(n_samples, cluster_size), 
                                        replace=False)
        
        for i, idx in enumerate(sample_indices):
            sample_text = df.iloc[idx][text_column]
            display_text = sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
            print(f"Sample {i+1}: {display_text}")