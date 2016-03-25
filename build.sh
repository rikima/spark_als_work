#!/bin/sh
cur=$(dirname $0)
pushd $cur

sbt clean package

popd