'''
这里用于解决服务器中证书错误问题，直接运行命令
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
'''


import os
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
# 替换下面的值为你的代理信息
os.environ["http_proxy"] = "http://用户名:密码@代理服务器地址:端口"
os.environ["https_proxy"] = "http://用户名:密码@代理服务器地址:端口"


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IndexTeam/IndexTTS-2",
    local_dir="checkpoints",
    local_dir_use_symlinks=False,   # 强制复制，不生成软链
)






