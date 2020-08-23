#!/usr/bin/env bash
echo -e "\033[33m[*] Start clean ! \033[0m"
cp -r runs/* bak_runs/
echo -e "[*] runs done "
cp -r checkpoint bak_runs/
echo -e "[*] checkpoint done "
rm -rf runs/*
rm -rf checkpoint/*
echo -e "\033[33m[*] Push Done ! \033[0m"
