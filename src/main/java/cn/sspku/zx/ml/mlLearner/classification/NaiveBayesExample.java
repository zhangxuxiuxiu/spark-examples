package cn.sspku.zx.ml.mlLearner.classification;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

public class NaiveBayesExample {

	public static void main(String[] args) {
		String wordFile = "file:///Users/zhangxu/Study/Spark/installation/spark-1.1.0/README.md";
		SparkConf conf = new SparkConf().setAppName("TF-IDF");
		JavaSparkContext sc = new JavaSparkContext(conf);
		@SuppressWarnings("serial")
		JavaRDD<LabeledPoint> training = sc.textFile(wordFile).map(
				new Function<String, LabeledPoint>() {

					public LabeledPoint call(String arg0) throws Exception {
						return new LabeledPoint(0, null);
					}
				}); // training set
		@SuppressWarnings("serial")
		JavaRDD<LabeledPoint> test = sc.textFile(wordFile).map(
				new Function<String, LabeledPoint>() {

					public LabeledPoint call(String arg0) throws Exception {
						return new LabeledPoint(0, null);
					}
				}); // test set

		final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);

		@SuppressWarnings("serial")
		JavaPairRDD<Double, Double> predictionAndLabel = test
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					public Tuple2<Double, Double> call(LabeledPoint p)
							throws Exception {
						return new Tuple2<Double, Double>(model.predict(p
								.features()), p.label());
					}

				});
		@SuppressWarnings("serial")
		double accuracy = 1.0
				* predictionAndLabel.filter(
						new Function<Tuple2<Double, Double>, Boolean>() {
							public Boolean call(Tuple2<Double, Double> pl)
									throws Exception {
								return pl._1() == pl._2();
							}

						}).count() / test.count();
		System.out.println(accuracy);
	}
}
