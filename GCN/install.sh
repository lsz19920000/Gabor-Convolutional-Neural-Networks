#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Install CPU version..."
cd $DIR/gcn
rm -r ./build
luarocks make gcn-scm-1.rockspec

echo "Install GPU version..."
cd $DIR/cugcn
rm -r ./build
luarocks make cugcn-scm-1.rockspec

echo "Install CuDNN version..."
cd $DIR/cudnngcn
rm -r ./build
luarocks make cudnngcn-scm-1.rockspec

echo "-------------------------------"
echo "All done!"

