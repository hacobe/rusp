#!/bin/bash
export WMT_DIR="wmt/"

wget http://data.statmt.org/wmt16/translation-task/test.tgz -P $WMT_DIR
tar -xvzf $WMT_DIR/test.tgz -C $WMT_DIR
mv $WMT_DIR/test $WMT_DIR/sgm16
rm $WMT_DIR/test.tgz

wget http://data.statmt.org/wmt17/translation-task/test.tgz -P $WMT_DIR
tar -xvzf $WMT_DIR/test.tgz -C $WMT_DIR
mv $WMT_DIR/test $WMT_DIR/sgm17
rm $WMT_DIR/test.tgz

wget http://data.statmt.org/wmt18/translation-task/test.tgz -P $WMT_DIR
tar -xvzf $WMT_DIR/test.tgz -C $WMT_DIR
mv $WMT_DIR/test $WMT_DIR/sgm18
rm $WMT_DIR/test.tgz

wget http://data.statmt.org/wmt19/translation-task/test.tgz -P $WMT_DIR
tar -xvzf $WMT_DIR/test.tgz -C $WMT_DIR
mv $WMT_DIR/sgm $WMT_DIR/sgm19
rm $WMT_DIR/test.tgz

wget http://data.statmt.org/wmt20/translation-task/test.tgz -P $WMT_DIR
tar -xvzf $WMT_DIR/test.tgz -C $WMT_DIR
mv $WMT_DIR/sgm $WMT_DIR/sgm20
rm $WMT_DIR/test.tgz