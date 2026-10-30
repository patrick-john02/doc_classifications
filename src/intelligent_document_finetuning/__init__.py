"""intelligent_document_finetuning: reusable ML code for the PAO archiving thesis.

Subpackages (one per task):
    ocr/            turn scanned images into text (calls the Intelligent-Document-Search backend)
    data/           load and clean datasets (RVL-CDIP, PAO metadata) into (text, label) tables
    topic_modeling/ unsupervised discovery of document groups (BERTopic)
    classification/ supervised document-type models (TF-IDF + LR baseline, SetFit)
    evaluation/     shared metrics and cross-validation helpers

Scripts in scripts/ import from here; this package never imports from scripts/.
"""
