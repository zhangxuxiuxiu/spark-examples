package cn.sspku.zx.ml.mlLearner;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

/**
 * Hello world!
 * 
 */
public class App {
	public static void main(String[] args) {
		// Should be some file on your system
		String logFile = "file:///Users/zhangxu/Study/Spark/installation/spark-1.1.0/README.md";
		SparkConf conf = new SparkConf().setAppName("Simple Application");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<String> logData = sc.textFile(logFile).cache();

		@SuppressWarnings("serial")
		long numAs = logData.filter(new Function<String, Boolean>() {
			public Boolean call(String s) {
				return s.contains("a");
			}
		}).count();

		@SuppressWarnings("serial")
		long numBs = logData.filter(new Function<String, Boolean>() {
			public Boolean call(String s) {
				return s.contains("b");
			}
		}).count();

		System.out.println("Lines with a: " + numAs + ", lines with b: "
				+ numBs);

	
		
	}
}
