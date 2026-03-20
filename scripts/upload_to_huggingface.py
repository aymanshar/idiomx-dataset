from huggingface_hub import create_repo, upload_file

repo_id = "AymanAliSharara/IdiomX"
create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

files = [
    ("data/final/idiomx_core.parquet", "idiomx_core.parquet"),
    ("data/final/idiomx_human_examples_only.parquet", "idiomx_human_examples_only.parquet"),
    ("data/final/dataset_statistics.json", "dataset_statistics.json"),
    ("data/final/source_distribution.csv", "source_distribution.csv"),
    ("README_HF.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("DATASET_LICENSE.md", "DATASET_LICENSE.md"),
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
]

for local_path, path_in_repo in files:
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )