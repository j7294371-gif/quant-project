FROM python:3.11-slim

WORKDIR /app

# 创建持久化目录（volume 挂载点）
RUN mkdir -p logs data state

# 分层构建：先安装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码（.dockerignore 排除敏感/缓存文件）
COPY . .

# 默认命令：打印帮助信息（docker-compose 中 command 会覆盖）
CMD ["python3", "main.py", "--help"]
