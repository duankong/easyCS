#!/bin/bash
# date:2019.08.17
# version:v1
# usage:
# 先复制/usr/local/nginx为例子  复制到/tmp下  不复制nginx中conf 和log logs文件夹
# 首先cd /usr/local
# 接着执行bash  t.5.sh nginx
 
 
if [ $# -lt 2 ]
  then echo '请填写路径'
  exit
fi

mkdir $2
 
{
  function e(){
    for file in `ls $1`
    do
      echo "$1/$file"
      if [ -d $1"/"$file ]
      then
        if [[ `echo $file | grep -e __pycache__` ]]||[[ `echo $file | grep -e checkpoint` ]]||[[ `echo $file | grep -e 17782` ]]
        then
          continue
          else
            mkdir -p $2/$file
        fi
      e $1"/"$file $2"/"$file
      else
        cp $1/$file $2/$file
      fi
    done
  }
}
e $1 $2
echo  "[*] CP Done!"