@echo off
title Build Documentation

echo Building EXOSIMS Documentation with sphinx

sphinx-apidoc -f -o . ../EXOSIMS/

del modules.rst

make html
make html
