#!/bin/sh
cur=$(dirname $0)

program=com.rikima.ml.recommend.SparkAlsPredictor

jar="$cur/target/scala-2.10/spark_als_work_2.10-0.1-SNAPSHOT.jar"

../spark/bin/spark-submit --class $program --master local[2] $jar $*