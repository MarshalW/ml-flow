#!/bin/bash

# 检查并安装unzip（自动判断root权限）
install_unzip() {
    local install_cmd
    if ! command -v unzip &>/dev/null; then
        echo "unzip未安装，正在安装..."

        # 判断当前是否是root用户
        if [ "$(id -u)" -eq 0 ]; then
            echo "当前是root用户，直接执行安装命令"
            install_cmd="apt-get update && apt-get install unzip -y"
        else
            echo "当前是非root用户，使用sudo执行安装命令"
            install_cmd="sudo apt-get update && sudo apt-get install unzip -y"
        fi

        # 执行安装命令
        if eval "$install_cmd"; then
            echo "unzip安装成功"
        else
            echo "unzip安装失败，请检查网络或权限" >&2
            exit 1
        fi
    else
        echo "unzip已安装，跳过安装步骤"
    fi
}

# 检查并安装ossutil（自动判断root权限）
install_ossutil() {
    # 下载并解压ossutil
    cd /tmp || exit 1
    curl -o ossutil-2.1.1-linux-amd64.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.1/ossutil-2.1.1-linux-amd64.zip
    unzip ossutil-2.1.1-linux-amd64.zip
    cd ossutil-2.1.1-linux-amd64 || exit 1
    chmod 755 ossutil

    # 判断当前是否是root用户
    if [ "$(id -u)" -eq 0 ]; then
        echo "当前是root用户，直接执行安装命令"
        mv ossutil /usr/local/bin/ && ln -s /usr/local/bin/ossutil /usr/bin/ossutil
    else
        echo "当前是非root用户，使用sudo执行安装命令"
        sudo mv ossutil /usr/local/bin/ && sudo ln -s /usr/local/bin/ossutil /usr/bin/ossutil
    fi

    # 验证安装是否成功
    if ossutil version &>/dev/null; then
        echo "ossutil安装成功！"
    else
        echo "ossutil安装失败，请检查权限或网络连接" >&2
        exit 1
    fi
}

# 主流程
install_unzip
install_ossutil

