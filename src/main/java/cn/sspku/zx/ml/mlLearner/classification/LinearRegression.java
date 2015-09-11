package cn.sspku.zx.ml.mlLearner.classification;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import scala.Tuple2;

public class LinearRegression {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf()
				.setAppName("Linear Regression Example");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load and parse the data
		String path = "data/mllib/ridge-data/lpsa.data";
		JavaRDD<String> data = sc.textFile(path);
		@SuppressWarnings("serial")
		JavaRDD<LabeledPoint> parsedData = data
				.map(new Function<String, LabeledPoint>() {
					public LabeledPoint call(String line) {
						String[] parts = line.split(",");
						String[] features = parts[1].split(" ");
						double[] v = new double[features.length];
						for (int i = 0; i < features.length - 1; i++)
							v[i] = Double.parseDouble(features[i]);
						return new LabeledPoint(Double.parseDouble(parts[0]),
								Vectors.dense(v));
					}
				});

		// Building the model
		int numIterations = 100;
		final LinearRegressionModel model = LinearRegressionWithSGD.train(
				JavaRDD.toRDD(parsedData), numIterations);

		// Evaluate model on training examples and compute training error
		@SuppressWarnings("serial")
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds = parsedData
				.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
					public Tuple2<Double, Double> call(LabeledPoint point) {
						double prediction = model.predict(point.features());
						return new Tuple2<Double, Double>(prediction, point
								.label());
					}
				});

		@SuppressWarnings("serial")
		Double MSE = new JavaDoubleRDD(valuesAndPreds.map(
				new Function<Tuple2<Double, Double>, Object>() {
					public Object call(Tuple2<Double, Double> pair) {
						return Math.pow(pair._1() - pair._2(), 2.0);
					}
				}).rdd()).mean();

		System.out.println("training Mean Squared Error = " + MSE);
	}
}
