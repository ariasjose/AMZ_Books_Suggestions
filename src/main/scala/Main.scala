  /**
   * Created by jarias on 4/4/2018.
  **/

  import org.apache.hadoop.fs.{FileSystem, Path}
  import org.apache.spark.ml.feature.{StringIndexer}
  import org.apache.spark.ml.{Pipeline}
  import org.apache.spark.mllib.recommendation.{ALS, Rating}
  import org.apache.spark.sql.SQLContext
  import org.apache.spark.SparkConf
  import org.apache.spark.SparkContext

  object Main {

    def main(args: Array[String]): Unit = {

      val conf = new SparkConf().setAppName("AMZBooksSE")
                  //.setMaster("yarn-client")

      val sc = new SparkContext(conf)
      val sqlContext = new SQLContext(sc)


      //Import books ratings (userId, productId, rating) - user and product ids are alphanumeric
      val booksRdd = sc.textFile(path="/amz/ratingsBooks.csv").map( r => {
          val values = r.split(',')
          (values(0), values(1), values(2))
      })

      //Convert to DF
      import sqlContext.implicits._
      val booksDF = booksRdd.toDF(colNames = "userId", "productId", "rating")

      //Create string indexers to map alphanumeric to Int as Rating class expects Int
      //types for user and product ids
      val strIndexerForUser = new StringIndexer().setInputCol("userId").setOutputCol("userIdInt")
      val strIndexerForProduct = new StringIndexer().setInputCol("productId").setOutputCol("productIdInt")

      val pipeline = new Pipeline().setStages(Array(strIndexerForUser, strIndexerForProduct))
      val pipelineModel = pipeline.fit(booksDF)

      //Save the model generated
      pipelineModel.save(path="/amz/pipelineModel")

      val booksDFWithIdsMapped = pipelineModel.transform(booksDF)
      booksDFWithIdsMapped show 5

      //+--------------+----------+------+---------+------------+
      //|        userId| productId|rating|userIdInt|productIdInt|
      //+--------------+----------+------+---------+------------+
      //|A3CW0ZLUO5X2B1|1439171300|   1.0|   6092.0|     32158.0|
      //|A2D7B5I7ZQ51XL|1439171300|   3.0|  21753.0|     32158.0|
      //|A34A7QEBMYTALW|1439171300|   1.0|  26053.0|     32158.0|
      //|A3CA3RWZYJDWXE|1439171300|   3.0|   3061.0|     32158.0|
      //|A2F6N60Z96CAJI|1439171300|   5.0|      2.0|     32158.0|
      //+--------------+----------+------+---------+------------+


      //Create rdd of Rating from DF and persist it as we'll use it later
      val booksRatingsRdd = booksDFWithIdsMapped.map( r => {
        Rating(
          r.getAs[Double]("userIdInt").toInt,
          r.getAs[Double]("productIdInt").toInt,
          r.getAs[String]("rating").toDouble
        )
      }).cache

      //Display data description
      val totalOfReviews = booksRatingsRdd.count
      val totalReviewers = booksRatingsRdd.map(_.user).distinct.count
      val totalBooks = booksRatingsRdd.map(_.product).distinct.count
      println(s"Total Reviews: $totalOfReviews; Total Reviewers: $totalReviewers; Books rated: $totalBooks")

      //Split ratings RDD: training data = 80% ;  test data = 20%
      val ratingsSplit = booksRatingsRdd.randomSplit(weights=Array(0.8, 0.2), seed=0L)
      val trainingDataRdd = ratingsSplit(0).cache
      val testDataRdd = ratingsSplit(1).cache

      //Display training and test data
      val trainingSize = trainingDataRdd.count
      val testSize = testDataRdd.count
      println(s"Training size: $trainingSize; Test size: $testSize")

      //Train and create the matrix factorization model
      val _rank = 5
      val iterations = 20
      var lambda = 0.01
      val matrixModel = ALS.train(trainingDataRdd, _rank, iterations, lambda)

      //Save the model for future usage
      val fs = FileSystem.get(sc.hadoopConfiguration)
      val path = new Path("/amz/alsMatrix")
      if(fs.exists(path)){
        fs.delete(path, true)
      }
      matrixModel.save(sc, path.toString)

      //Make predictions using the model for the test data
      val predictionsForTestData = matrixModel.predict(
          testDataRdd.map(r => (r.user, r.product))
      )

      //Validate our model by comparing predicted and actual ratings for test data users
      //Will use join to merge our results and then validate our model using Mean Absolute Error
      val predictedRatingsRdd = predictionsForTestData.map( r => ((r.user, r.product), r.rating))
      val testDataRatingsRdd = testDataRdd.map( r => ((r.user, r.product), r.rating))

      //(user, product), (ratingP, ratingT)
      val predictedAndActualRatingsRdd = predictedRatingsRdd.join(testDataRatingsRdd)

      //Get the MAE and display it - The lower the MAE the better the model
      val mae = predictedAndActualRatingsRdd.map({
          case ((user, product), (ratingP, ratingT)) => math.abs(ratingP - ratingT)
      }).mean
      println(s"MAE for our model: $mae")

      //Calculate MAE removing false positives
      val predictedAndActualRatingsRddWithoutFP = predictedAndActualRatingsRdd.filter({
        case ((user, product), (ratingP, ratingT)) => ratingT > 1 || ratingP < 5
      })
      val maeNoFP = predictedAndActualRatingsRddWithoutFP.map({
        case ((user, product), (ratingP, ratingT)) => math.abs(ratingP - ratingT)
      }).mean
      println(s"MAE for our model removing false positives: $maeNoFP")

      //Get the actual mapping for users and products
      val usersMap = booksDFWithIdsMapped.map( r => {
        (r.getAs[Double]("userIdInt").toInt, r.getAs[String]("userId"))
      }).distinct.collectAsMap

      //As we haven't loaded the products let's create a Rdd of products (id, title)
      val productsRdd = sqlContext.read.json(path="/amz/metaBooks.json").map( r => {
        (r.getAs[String]("asin"), r.getAs[String]("title"))
      })

      //(id, idInt) join (id, title) => (id, (idInt, Title)) => (idInt, (idTitle))
      val productsMap = booksDFWithIdsMapped.map( r => {
          (r.getAs[String]("productId"), r.getAs[Double]("productIdInt").toInt)
      }).distinct.join(productsRdd).map({
          case(productId, (productIdInt, title)) => (productIdInt, (productId, title))
      }).collectAsMap()

      //Let's choose a random user and give them suggestions
      //Get random user
      val userIdInt = booksRatingsRdd.takeSample(withReplacement=false, num=1)(0).user

      //Display user's products and ratings
      val userId = usersMap.get(userIdInt).get
      println(s"Reviewed products by user $userId:")
      println("Title | Rating")
      booksRatingsRdd.filter(_.user == userIdInt).map(r => {
        (r.product, r.rating)
      }).collect().foreach(r => {
        println(productsMap.get(r._1).get._2 + " | " + r._2)
      })

      //Display recommendations
      val topFiveRecommendations = matrixModel.recommendProducts(userIdInt, 10)
      println(s"Top 10 recommendations for user $userId:")
      println("Title")
      topFiveRecommendations.foreach(r => {
        println(productsMap.get(r.product).get._2)
      })
    }
  }
