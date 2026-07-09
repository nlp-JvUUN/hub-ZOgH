#!/bin/bash
# =============================================================================
# TopWAF 知识助手 - 一键部署脚本（Ubuntu / Debian）
# 用法：bash deploy/deploy.sh
# =============================================================================

set -e

APP_DIR="/opt/wyquestion"
APP_USER="wyquestion"
APP_PORT="7321"

echo "=== 1/6 创建系统用户 ==="
if id "$APP_USER" &>/dev/null; then
    echo "用户 $APP_USER 已存在，跳过"
else
    useradd -r -m -s /bin/false "$APP_USER"
    echo "创建用户 $APP_USER 成功"
fi

echo ""
echo "=== 2/6 创建目录 ==="
mkdir -p "$APP_DIR"
echo "目录已创建: $APP_DIR"

echo ""
echo "=== 3/6 复制项目文件 ==="
# 排除敏感文件和临时文件
rsync -av --exclude='__pycache__' \
           --exclude='.git' \
           --exclude='.mypy_cache' \
           --exclude='.vscode' \
           --exclude='kb/' \
           --exclude='dist/' \
           . "$APP_DIR/"
echo "文件同步完成"

echo ""
echo "=== 4/6 安装 Python 依赖 ==="
pip install -r "$APP_DIR/deploy/requirements.txt" -q
echo "依赖安装完成"

echo ""
echo "=== 5/6 配置环境变量 ==="
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/deploy/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo "请编辑 $APP_DIR/.env 填入真实 API Key"
else
    echo ".env 已存在，跳过"
fi

echo ""
echo "=== 6/6 配置 systemd 服务 ==="
cp "$APP_DIR/deploy/systemd/wyquestion.service" /etc/systemd/system/
chown root:root /etc/systemd/system/wyquestion.service
systemctl daemon-reload
systemctl enable wyquestion
systemctl restart wyquestion
echo "服务已启动: wyquestion"

echo ""
echo "=== 部署完成 ==="
echo "服务地址: http://127.0.0.1:$APP_PORT"
echo ""
echo "常用命令："
echo "  查看状态:  systemctl status wyquestion"
echo "  查看日志:  journalctl -u wyquestion -f"
echo "  重启服务:  systemctl restart wyquestion"
echo "  停止服务:  systemctl stop wyquestion"
