'''
下载大模型文件
默认路径在~/.cache//huggingface/hub/
'''
from huggingface_hub import snapshot_download
model_name = "Skywork/Skywork-13B-Base"
snapshot_download(repo_id=model_name)
