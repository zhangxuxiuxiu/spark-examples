package cn.sspku.zx.ml.mlLearner;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;

public class TFIDF {
	public static void main(String[] args) {
		String wordFile = "file:///Users/zhangxu/Study/Spark/installation/spark-1.1.0/README.md";
		SparkConf conf = new SparkConf().setAppName("TF-IDF");
		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<List<String>> wordData = sc.textFile(wordFile)
				.map(new Function<String, List<String>>() {
					/**
					 * 
					 */
					private static final long serialVersionUID = 1091920418241245797L;

					public List<String> call(String line) throws Exception {
						String[] words = line.split(" ");
						return Arrays.asList(words);
					}

				}).cache();
		HashingTF hTF = new HashingTF();
		JavaRDD<Vector> tf = hTF.transform(wordData).cache();
		IDFModel idfModel = new IDF().fit(tf);
		JavaRDD<Vector> tfidf = idfModel.transform(tf);
		System.out.println(tfidf);
	}
}
