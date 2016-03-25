package com.rikima.ml.recommend

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

/**
  * Created by masakir on 2016/03/21.
  */
object SparkAlsTrainer {

  def execute(sc: SparkContext, input: String, rank: Int = 10, numIterations: Int = 10, alpha: Double = 0.01): Unit = {
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
    val model = ALS.train(ratings, rank, numIterations, alpha)

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

    // Save and load model
    model.save(sc, input + ".model")
  }


  def main(args: Array[String]): Unit = {
    var input = ""
    var rank = 10
    var numIterations = 10
    var alpha = 0.01
    for (i <- 0 until args.length) {
      val a = args(i)
      if (a == "-i" || a == "--input") {
        input = args(i+1)
      }
      if (a == "-k" || a == "--rank") {
        rank = args(i+1).toInt
      }
      if (a == "-t" || a == "--num_iterations") {
        numIterations = args(i+1).toInt
      }
      if (a == "-a" || a == "--alpha") {
        alpha = args(i+1).toDouble
      }
    }

    val sc = new SparkContext()

    execute(sc, input, rank, numIterations, alpha)
  }
}
