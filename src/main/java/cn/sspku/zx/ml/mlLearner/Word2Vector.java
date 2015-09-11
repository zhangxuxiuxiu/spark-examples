package cn.sspku.zx.ml.mlLearner;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;

import scala.Tuple2;

public class Word2Vector {

	public static void main(String[] args) {
		String wordFile = "file:///Users/zhangxu/Study/Spark/installation/spark-1.1.0/README.md";
		SparkConf conf = new SparkConf().setAppName("Word2Vec");
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

		Word2Vec word2vec = new Word2Vec();

		Word2VecModel model = word2vec.fit(wordData);

		Tuple2<String, Object>[] synonyms = model.findSynonyms("china", 40);
		for (int idx = 0; idx < synonyms.length; ++idx)
			System.out.println("synonym=" + synonyms[idx]._1 + "   cosSim="
					+ synonyms[idx]._2);
	}
}
