// Databricks notebook source
spark

// COMMAND ----------

// Import required Spark libraries

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import sys.process._

// COMMAND ----------

// Read contents of "ratings.dat" and show sample content

// COMMAND ----------

val ratings_raw = sc.textFile("/FileStore/tables/ratings.dat")
ratings_raw.takeSample(false,10, seed=0).foreach(println)

// COMMAND ----------

// Convert ratings data to a DataFrame

// COMMAND ----------

case class Rating(userId: Int, movieId: Int, rating: Float)
val ratings = ratings_raw.map(x => x.split("::")).map(r => Rating(r(0).toInt, r(1).toInt, r(2).toFloat)).toDF().na.drop()

// COMMAND ----------

// Show the number of ratings in the dataset is slightly more than one million

// COMMAND ----------

println("Number of ratings = " + ratings_raw.count())

// COMMAND ----------

// Show a sample of the Ratings DataFrame

// COMMAND ----------

ratings.sample(false, 0.0001, seed=0).show(10)

// COMMAND ----------

// Show sample number of ratings per user

// COMMAND ----------

val grouped_ratings = ratings.groupBy("userId").count().withColumnRenamed("count", "No. of ratings")
grouped_ratings.show(10)

// COMMAND ----------

// Show the number of users in the dataset is approximately 6000

// COMMAND ----------

println("Number of users = " + grouped_ratings.count())

// COMMAND ----------

// Movies File Description

// Movie information is in the file “movies.dat” and is in the following format:

//   movieId::Title::Genres


// COMMAND ----------

val movies_raw = sc.textFile("/FileStore/tables/movies.dat")
movies_raw.takeSample(false,10, seed=0).foreach(println)

// COMMAND ----------

// Convert movies data to a DataFrame

// COMMAND ----------

case class Movies(movieId: Int, Title: String, Genre: String)
val movies = movies_raw.map(x => x.split("::")).map(m => Movies(m(0).toInt, m(1).toString, m(2).toString )).toDF()
movies.show(10, false)

// COMMAND ----------

// Show the number of movies in the dataset is approximately 4000

// COMMAND ----------

println("Number of movies = " + movies.count())

// COMMAND ----------

// Split Ratings data into Training (80%) and Test (20%) datasets

// COMMAND ----------

val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2), seed=0L)

// COMMAND ----------

// Show resulting Ratings dataset counts

// COMMAND ----------

val trainingRatio = training.count().toDouble/ratings.count().toDouble*100
val testRatio = test.count().toDouble/ratings.count().toDouble*100

// COMMAND ----------

println("Total number of ratings = " + ratings.count())
println("Training dataset count = " + training.count() + ", " + BigDecimal(trainingRatio).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble + "%")
println("Test dataset count = " + test.count() + ", " + BigDecimal(testRatio).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble+ "%")

// COMMAND ----------

// Show sample of Ratings Training dataset

// COMMAND ----------

training.sample(false, 0.0001, seed=0).show(10)

// COMMAND ----------

// Show sample of Ratings Test dataset

// COMMAND ----------

test.sample(false, 0.0001, seed=0).show(10)

// COMMAND ----------

// Build the recommendation model on the training data using ALS

// COMMAND ----------

val als = new ALS().setMaxIter(10).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = als.fit(training)

// COMMAND ----------

println(als.explainParams)

// COMMAND ----------

// Run the model against the Test data and show a sample of the predictions

// COMMAND ----------

val predictions = model.transform(test).na.drop()
predictions.show(10)

// COMMAND ----------

// Evaluate the model by computing the RMSE on the test data

// COMMAND ----------

val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println("Root-mean-square error = " + rmse)

// COMMAND ----------

// Show that a smaller value of rmse is better

// COMMAND ----------

evaluator.isLargerBetter

// COMMAND ----------

// Tune the Model

// COMMAND ----------

val paramGrid = new ParamGridBuilder().addGrid(als.regParam, Array(0.01, 0.1)).build()

// COMMAND ----------

// Create a cross validator to tune the model with the defined parameter grid

// COMMAND ----------

val cv = new CrossValidator().setEstimator(als).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)

// COMMAND ----------

cv.getEstimatorParamMaps.foreach(println)

// COMMAND ----------

// Cross-evaluate to find the best model

// COMMAND ----------

val cvModel = cv.fit(training)
println("Best fit root-mean-square error = " + evaluator.evaluate(cvModel.transform(test).na.drop()))

// COMMAND ----------

// Now lets use our model to recommend movies to a user

// COMMAND ----------

val userId = 3000

// COMMAND ----------

// Create a DataFrame with the movies that user 3000 has rated

// COMMAND ----------

val movies_watched = ratings.filter(ratings("userId") === userId)
movies_watched.show(10)

// COMMAND ----------

// Calculate user 3000's minimum, maximum and average movie rating

// COMMAND ----------

movies_watched.select(min($"rating"), max($"rating"), avg($"rating") ).show()

// COMMAND ----------

// Show user 3000's top 10 rated movies (with movie title, genre, and rating)

// COMMAND ----------

ratings.as("a").filter(ratings("userId") === userId).join(movies.as("b"), $"a.movieId" === $"b.movieId").select("a.userId", "a.movieId", "b.Title", "b.Genre",  "a.rating").sort($"a.rating".desc).show(10,false)

// COMMAND ----------

// Determining what movies user 3000 has not already watched and rated so that we can make new movie recommendations

// COMMAND ----------

val movies_notwatched = ratings.filter(test("userId") =!= userId)
movies_notwatched.sample(false, 0.0001, seed=0).show(5)
println("Count = " + movies_notwatched.count())

// COMMAND ----------

val r1 = movies_notwatched.select("movieId").distinct
val r2 = r1.withColumn("userId", lit(userId))

// COMMAND ----------

println("Number of movies NOT rated by user = " + r2.count())

// COMMAND ----------

val data_userId1 = movies_notwatched_movieId.withColumn("userId", lit(userId))
val data_userId = data_userId1.select("userId","movieId")
data_userId.orderBy($"movieId".desc).show()

// COMMAND ----------

// Creating movie recommendations for userId = 3000 

// COMMAND ----------

val predictions_userId = cvModel.transform(r2).na.drop()
val df = predictions_userId.as("t").join(movies.as("m"), $"t.movieId" === $"m.movieId")

// COMMAND ----------

// Top 10 recommended movies 

// COMMAND ----------

val top10 = df.select("Title", "prediction").sort($"prediction".desc).show(10,false)
