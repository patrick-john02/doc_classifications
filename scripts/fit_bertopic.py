#Fit BERTopic on the RVL-CDIP OCR corpus.

#MiniLM -> UMAP -> HDBScan -> c-TF-IDF. Labels are NOT used for fitting, 
#only for the quick sanity score printed at the end.
#Run: python scripts/fit_bertopic.py --run-name v1_default

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP




ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "rvl_cdip_v1.parquet"
EMB_CACHE = ROOT / "data" / "interim" / "rvl_cdip_v1_minilm.npy"
MODELS = ROOT / "models" / "bertopic"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CUSTOM_STOPWORDS = {
    "philip", "morris", "lorillard", "kent", "rjr", "reynolds", "brown", "williamson",
    "tif", "lal", "het", "van", "boe", "nre", "brans", "rou", "inh",
}

def load_corpus()->pd.DataFrame:
    df = pd.read_parquet(CORPUS)
    df = df[df["usable"]].reset_index(drop=True)
    print(f"[data] {len(df)} usable docs, {df['label'].nunique()} true classes (hidden from model)")
    return df

def load_or_embed(df:pd.DataFrame, encoder: SentenceTransformer)->np.ndarray: 
    ids_file = EMB_CACHE.with_suffix(".ids.txt")
    ids = df["doc_id"].astype(str).tolist()
    if EMB_CACHE.exists() and ids_file.exists() and ids_file.read_text().split("\n") == ids:
        print("[embed] using cached embeddings")
        return np.load(EMB_CACHE)

    print(f"[embed] encoding {EMB_CACHE}")
    emb = encoder.encode(df["text_clean"].tolist(), batch_size=64, show_progress_bar=True)
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_CACHE, emb)
    ids_file.write_text("\n".join(ids))
    return emb


def build_model(args, encoder)->BERTopic:
    umap_model = UMAP(n_neighbors=args.n_neighbors, n_components=5, min_dist=0.0, metric="cosine", random_state=args.seed)
    
    hdbscan_model = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                            metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    
    vectorizer = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS | CUSTOM_STOPWORDS), min_df=2, ngram_range=(1,2),
                                 token_pattern=r"(?u)\b[a-zA-Z]{3,}\b") 
    
    return BERTopic(embedding_model=encoder, umap_model=umap_model,
                    hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
                    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                    top_n_words=10, verbose=True)


def sanity_scores(labels: pd.Series, topics:np.ndarray)->dict:
    mask = topics != -1
    return{
        "n_topics": int(len(set(topics)) - (1 if -1 in topics else 0)),
        "outlier_pct":round(float((~mask).mean() * 100), 1),
        "ari_all": round(adjusted_rand_score(labels, topics), 3),
        "nmi_all":round(normalized_mutual_info_score(labels, topics), 3),
        "ari_clustered": round(adjusted_rand_score(labels[mask], topics[mask]), 3),
        "nmi_clustered": round(normalized_mutual_info_score(labels[mask], topics[mask]), 3),
    }

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-name", default="v1_default")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df=load_corpus()
    encoder = SentenceTransformer(EMBED_MODEL)
    embeddings = load_or_embed(df, encoder)

    topic_model = build_model(args, encoder)
    topics, _  = topic_model.fit_transform(df["text_clean"].tolist(), embeddings)
    topics = np.asarray(topics)

    out_dir=MODELS / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(out_dir / "model", serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=EMBED_MODEL)

    df.assign(topic=topics)[["doc_id", "label", "topic", "ocr_score","chars"]] \
    .to_parquet(out_dir / "assignments.parquet", index=False)
    topic_model.get_topic_info().drop(columns=["Representative_Docs"], errors="ignore") \
    .to_csv(out_dir / "topic_info.csv",  index=False)

    scores = sanity_scores(df["label"], topics)
    (out_dir / "run.json").write_text(json.dumps({"params" : vars(args), "scores": scores}, indent=2))

    print("\n[topics]")
    print(topic_model.get_topic_info()[["Topic", "Count", "Name"]].head(25).to_string(index=False))
    print("\n[Sanity vs hidden labels]", scores)
    print("\n[dominant true label per topic]")
    ct = pd.crosstab(topics, df["label"])
    summary = pd.DataFrame({"size":ct.sum(1), "top_label": ct.idxmax(1)})
    print(summary.to_string())
    print(f"\nSaved to {out_dir}")

if __name__ == "__main__":
    main()


