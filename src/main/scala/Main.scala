  /**
    * Created by jarias on 4/4/2018.
    */

  import org.apache.hadoop.fs.{FileSystem, Path}
  import org.apache.spark.ml.feature.{StringIndexer}
  import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
  import org.apache.spark.ml.{Pipeline}
  import org.apache.spark.sql.SQLContext
  import org.apache.spark.SparkConf
  import org.apache.spark.SparkContext

  object Main {

    //case class Product(Name:String)

    def main(args: Array[String]): Unit = {

      val conf = new SparkConf().setAppName("AMZVGSE")
                  //.setMaster("yarn-client")

      val sc = new SparkContext(conf)
      val sqlContext = new SQLContext(sc)


      //Import video games ratings (userId, productId, rating) - user and product ids are alphanumeric
      val videoGamesRDD = sc.textFile("file:///C:/Spark/data/AMZ/ratingsVideoGames.csv").map( r => {
          val values = r.split(',')
          (values(0), values(1), values(2))
      })

      //Convert to DF
      import sqlContext.implicits._
      val videoGamesDF = videoGamesRDD.toDF(colNames = "userId", "productId", "rating")

      //Create string indexers to map alphanumeric to Int as Rating class expects Int
      //types for user and product ids
      val strIndexerForUser = new StringIndexer().setInputCol("userId").setOutputCol("userIdInt")
      val strIndexerForProduct = new StringIndexer().setInputCol("productId").setOutputCol("productIdInt")

      val pipeline = new Pipeline().setStages(Array(strIndexerForUser, strIndexerForProduct))
      val pipelineModel = pipeline.fit(videoGamesDF)
      val videoGamesDFWithIdsMapped = pipelineModel.transform(videoGamesDF)

      //|userId|productId|rating|userIdInt|productIdInt
      //|||||
      //|||||

      //Create rdd of Rating from DF and persist it as we'll use it later
      val videoGamesRatingsRdd = videoGamesDFWithIdsMapped.map( r => {
          Rating(r.getAs[Double]("userIdInt").toInt,
                  r.getAs[Double]("productIdInt").toInt,
                  r.getAs[String]("rating").toDouble)
      }).cache

      //Display data description
      var totalOfReviews = videoGamesRatingsRdd.count
      var totalReviewers = videoGamesRatingsRdd.map(_.user).distinct.count
      var totalGames = videoGamesRatingsRdd.map(_.product).distinct.count
      println(s"Total Reviews: $totalOfReviews; Total Reviewers: $totalReviewers; Games rated: $totalGames")

      //Split ratings RDD: training data = 80% ;  test data = 20%
      val ratingsSplit = videoGamesRatingsRdd.randomSplit(weights=Array(0.8, 0.2), seed=0L)
      val trainingDataRdd = ratingsSplit(0).cache
      val testDataRdd = ratingsSplit(1).cache

      //Display training and test data
      val trainingSize = trainingDataRdd.count
      val testSize = testDataRdd.count
      println(s"Training size: $trainingSize; Test size: $testSize")


      //Train and create model
      val rank = 5
      val iterations = 20
      val matrixModel = ALS.train(trainingDataRdd, rank, iterations)

      //Make predictions using the model for the test data
      val predictionsForTestData = matrixModel.predict(
            testDataRdd.map(r => (r.user, r.product))
      )

      //Validate our model by comparing predicted and actual ratings for test data users
      //Will use join to merge our results and then validate our model using Mean Absolute Error
      val predictedRatingsRdd = predictionsForTestData.map( r => ((r.user, r.product), r.rating))
      val testDataRatingsRdd = testDataRdd.map( r => ((r.user, r.product), r.rating))
      val predictedAndActualRatingRdd = predictedRatingsRdd.join(testDataRatingsRdd)

      //Get the MAE amd display it - The lower the MAE the better the model
      val mae = predictedAndActualRatingRdd.map({
          case ((user, product), (ratingP, ratingT)) => math.abs(ratingP - ratingT)
      }).mean

      println(s"MAE for our model: $mae")

      //Get the actual mapping for users and products
      val usersMap = videoGamesDFWithIdsMapped.map( r => {
        (r.getAs[Double]("userIdInt").toInt, r.getAs[String]("userId"))
      }).distinct.collectAsMap

      val productsMap = videoGamesDFWithIdsMapped.map( r => {
        (r.getAs[Double]("productIdInt").toInt, r.getAs[String]("productId"))
      }).distinct.collectAsMap

      //Let's choose a random user
      val userIdInt = videoGamesRatingsRdd.takeSample(withReplacement=false, num=1)(0).user

      //Display user's products and ratings
      val userId = usersMap.get(userIdInt).get
      println(s"Reviewed products by user $userId:")
      videoGamesRatingsRdd.filter(_.user == userIdInt).map(r => {
        (r.product, r.rating)
      }).collect().foreach(r => {
        println(productsMap.get(r._1) + "\t" + r._2)
      })

      //Display recommendations
      val topFiveRecommendations = matrixModel.recommendProducts(userIdInt, 5)
      println(s"Top 5 recommendations for user $userId:")
      topFiveRecommendations.foreach(r => {
        println(productsMap.get(r.product) + "\t" + r.rating)
      })

    }
  }
