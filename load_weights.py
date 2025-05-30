from huggingface_hub import snapshot_download

save_dir = "./BAGEL-7B-MoT-DF11"
repo_id = "DFloat11/BAGEL-7B-MoT-DF11"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
)
