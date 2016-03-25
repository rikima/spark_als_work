package com.rikima.ml.recommend

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

/**
  * Created by masakir on 2016/03/21.
  */
object SparkAlsPredictor {

  def execute(sc: SparkContext, input: String, model_path: String): Unit = {
    // Load and parse the data
    val data = sc.textFile(input).map {
      case l =>
        val p = l.indexOf("#")
        l.substring(0, p)
    }
    val ratings = data.map(_.split('\t') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build the recommendation model using ALS
    val model = MatrixFactorizationModel.load(sc, model_path)

    // Evaluate the model on rating data
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println("Mean Squared Error = " + MSE)
  }


  def main(args: Array[String]): Unit = {
    var input = ""
    var model_path = ""
    for (i <- 0 until args.length) {
      val a = args(i)
      if (a == "-i" || a == "--input") {
        input = args(i+1)
      }
      if (a == "-m" || a == "--model") {
        model_path = args(i+1)
      }
    }
    val sc = new SparkContext()
    execute(sc, input, model_path)
  }
}
