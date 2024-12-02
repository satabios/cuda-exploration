#!/bin/bash

file_name="main"  # No spaces around the '='
if [ -f "$file_name" ]; then 
  rm $file_name 
fi
nvcc -arch=sm_80 $file_name.cu -o $file_name
echo -e " \t\t\t\t ----------Output----------"
./$file_name