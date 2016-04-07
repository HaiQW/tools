#!/bin/sh
#use  . proxy_wb.sh
http_proxy="http://root:root@10.214.16.213:808"
export http_proxy="http://root:root@10.214.16.213:808"
#export http_proxy="http://10.214.16.212:8888"
https_proxy="http://root:root@10.214.16.213:808"
export https_proxy="http://root:root@10.214.16.213:808"
#apt-get update
#export GIT_PROXY_COMMAND="/share/yc/git-proxy.sh"
export GIT_PROXY_COMMAND="/share/yc/git-proxy.sh"
#http_proxy="http://wb:wb@10.214.16.21:808"
#https_proxy="http://wb:wb@10.214.16.21:808"
