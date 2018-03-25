package com.walmart.spark


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.spark.mllib.recommendation._

object MovieRecALS {
   
  //**************** This method returns a Map of [MovieID -> MovieName] ********************
  def loadMovieNames() : Map[Int,String] = {
    
    //Handling the character encoding issues:
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)
    
    
    //Creating a Map of Int to Strings and populating it from u.item in the data source
    var movieNames : Map[Int, String] = Map()
    
    //Importing the data using scala.io.Source
    val lines = Source.fromFile("ml-100k/u.item").getLines()

    /*
     * The u.item data source contains names of the movies. 
     * Example:
     * 1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
     * So we need to extract only the first 2 relevant fields ie. (1)ID  and (2)Title, which are separated by '|'
     * Then we map them with each other
     */
    
    for(line <- lines){
      var fields = line.split("|")
      if(fields.length > 1){
        movieNames += (fields(0).toInt -> fields(1))
      }
       
      }
   return movieNames
 }
  
  def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Creating a Spark Context with Every core on the local Machine
    // Note:* When deploying the Driver program in a production environment, do not specify anything as it will
    //         override the cluster configuration. May lead to not utilizing the cluster capabilities.
    
    val sc = new SparkContext("local[*]","MovieRecALS")
    println("Now loading the movie names")
    val nameDict = loadMovieNames()
    
    // Now loading the actual data containing the movie ratings, using the SparkContext object 
    val data = sc.textFile("ml-100k/u.data")
    
    /*
     * ****** RATINGS DATA FORMAT(u.data) **********  
     *  UserID MovieID Rating 
     *  0 50 5 881250949
        0 172 5 881250949
        0 133 1	881250949
      196	242	3	881250949
     */
    
    val ratings = data.map(x=> x.split('\t')).map(x => Rating(x(0).toInt,x(1).toInt,x(2).toDouble)).cache()
    
    /*
     * This maps the data into Rating Objects which are a part of the MLLib Library
     *
     * 
     * The signature of a Rating object=>         
     * 
     * 					Rating(int user, int product, double rating)
     * 
     * 
     * Methods on a Rating object:
     *  abstract static boolean	canEqual(Object that) 
                abstract static boolean	equals(Object that) 
                int	product() 
                abstract static int	productArity() 
                abstract static Object	productElement(int n) 
                static scala.collection.Iterator<Object>	productIterator() 
                static String	productPrefix() 
                double	rating() 
          			int	user() 
     *     
     * Then we cache it. 
     */
    // 
    
    println("\nNow Training the recommendation model on the data")
    
    // The rank & numIterations have been selected as we have found them to be a good configuration for ALS
    // A lot of people have found good results with this config. Hence taking an informed guess 
    val rank = 8
    val numIterations = 20
   
    // Now Modeling the Rating data using ALS method for Collaborative Filtering 
    // Read more on: https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/
    // Used in recommender systems
    val model = ALS.train(ratings, rank, numIterations)

    // Taking UserID from commandline Arguments
    val userId = args(0).toInt
    
    println("\nRatings for UserID "+ userId + " :")
    
    val userRatings = ratings.filter(x => x.user == userId)
    val myRatings = userRatings.collect()
    
    // Printing the Ratings that I HARCODED FOR GETTING PERSONALIZED RECOMMENDATIONS for myself
    for(rating <- myRatings){
      println(nameDict(rating.product.toInt) + ": " + rating.rating.toString)     // Movie Names had been loaded as Map(like Python Dictionary) into nameDict
    }
    
    println("\n Printing top 10 recommendations as per the selections that I hardcoded:")
    
    // Generating top 10 recommendations for the given userID
    val recommendations = model.recommendProducts(userId, 10)
    
    for (recommendation <- recommendations){
      println(nameDict(recommendation .product.toInt) + " score" + recommendation.rating)
    }
    
    
    
    
  }
  
  
}