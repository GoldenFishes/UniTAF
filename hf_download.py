'''
这里用于解决5090服务器中直接运行命令
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
证书错误问题
'''


import os
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["http_proxy"] = "http://qiangong.zhou:goldeneagle1A@172.16.2.57:3128"
os.environ["https_proxy"] = "http://qiangong.zhou:goldeneagle1A@172.16.2.57:3128"


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IndexTeam/IndexTTS-2",
    local_dir="checkpoints",
    local_dir_use_symlinks=False,   # 强制复制，不生成软链
)






