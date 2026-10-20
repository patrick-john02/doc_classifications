# intelligent-document-finetuning
 PyTorch fine-tuning pipeline for domain-adapted LLMs and embeddings powering an intelligent multi-agent document archiving and RAG system.


python scripts/fit_bertopic.py --run-name v1_default

step 2
python scripts/fit_bertopic.py --run-name v2_mcs10 --min-cluster-size 10
python scripts/fit_bertopic.py --run-name v3_mcs25 --min-cluster-size 25
